# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# Maintainer: ndtrung

import importlib
from functools import partial
from typing import Callable
from warnings import warn

import lammps
from lammps import lammps
from lammps.dlext import (
    AccessLocation,
    AccessMode,
    DLExtSampler
)
from jax import jit
from jax import numpy as np
from jax.dlpack import from_dlpack as asarray

from pysages.backends.core import ContextWrapper
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.backends.snapshot import restore as _restore
from pysages.methods import SamplingMethod
from pysages.utils import check_device_array, copy

# TODO: Figure out a way to automatically tie the lifetime of Sampler
# objects to the contexts they bind to
CONTEXTS_SAMPLERS = {}

# need some API from context (context = lammps(cmdargs=args))
def is_on_gpu(context):
    on_gpu = False
    pkg_kokkos_enabled = context.has_package('KOKKOS')
    if pkg_kokkos_enabled == True:
        kokkos_backends = context.accelerator_config["KOKKOS"]
        if 'cuda' in kokkos_backends["api"]:
            on_gpu = True
    return on_gpu

# return a functor
def get_run_method(context, ntimesteps, **kwargs):
    def execute_run_cmd(ntimesteps, **kwargs):
      context.execute(f"run {ntimesteps}")
    return execute_run_cmd

def get_dimension(context):
    return context.extract_setting("dimension")

def get_timestep(context):
    return context.extract_global("dt")

def get_global_box(context):
    boxlo, boxhi, xy, yz, xz, periodicity, _ = context.extract_box()
    Lx = boxhi[0] - boxlo[0]
    Ly = boxhi[1] - boxlo[1]
    Lz = boxhi[2] - boxlo[2]
    origin = boxlo
    H = ((Lx, xy * Ly, xz * Lz), (0.0, Ly, yz * Lz), (0.0, 0.0, Lz))
    return H, origin

def set_post_force_hook(context, post_force_hook):
    # set_fix_external_callback
    context.set_fix_external_callback(post_force_hook)

# DLExtSampler is exported from lammps_dlext
class Sampler(DLExtSampler):
    def __init__(self, method_bundle, bias, callback: Callable, restore):
        initial_snapshot, initialize, method_update = method_bundle

        def update(positions, velocities, forces, tags, images, timestep):
            snapshot = self._pack_snapshot(positions, velocities, forces, tags, images)
            self.state = method_update(snapshot, self.state)
            self.bias(snapshot, self.state)
            if self.callback:
                self.callback(snapshot, self.state, timestep)

        super().__init__(update, default_location(), AccessMode.Read)
        self.state = initialize()
        self.callback = callback
        self.bias = bias
        self.box = initial_snapshot.box
        self.dt = initial_snapshot.dt
        self._restore = restore

    def restore(self, prev_snapshot):
        def restore_callback(positions, velocities, forces, tags, images, n):
            snapshot = self._pack_snapshot(positions, velocities, forces, tags, images)
            self._restore(snapshot, prev_snapshot)

        self.forward_data(restore_callback, default_location(), AccessMode.Overwrite, 0)

    def take_snapshot(self):
        container = []

        def snapshot_callback(positions, velocities, forces, tags, images, n):
            snapshot = self._pack_snapshot(positions, velocities, forces, tags, images)
            container.append(copy(snapshot))

        self.forward_data(snapshot_callback, default_location(), AccessMode.Read, 0)
        return container[0]

    def _pack_snapshot(self, positions, velocities, forces, tags, images):
        return Snapshot(
            asarray(positions),
            asarray(velocities),
            asarray(forces),
            asarray(tags),
            asarray(images),
            self.box,
            self.dt,
        )


if hasattr(AccessLocation, "OnDevice"):

    def default_location():
        return AccessLocation.OnDevice

else:

    def default_location():
        return AccessLocation.OnHost

# build a Snapshot object that contains all the tensors from the context
# 
def take_snapshot(wrapped_context, location=default_location()):
    context = wrapped_context.context

    # asarray needs argument of type DLManagedTensorPtr 
    positions =  copy(asarray(context.get_positions(location, AccessMode.Read)))
    types =      copy(asarray(context.get_types(location, AccessMode.Read)))
    velocities = copy(asarray(context.get_velocities(location, AccessMode.Read)))
    net_forces = copy(asarray(context.get_net_forces(location, AccessMode.ReadWrite)))
    tags =       copy(asarray(context.get_tags(location, AccessMode.Read)))
    imgs =       copy(asarray(context.get_images(location, AccessMode.Read)))

    #rtags = copy(asarray(context.get_rtags(context, location, AccessMode.Read)))

    check_device_array(positions)  # currently, we only support `DeviceArray`s

    H, origin = get_global_box(context)
    dt = get_timestep(context)

    return Snapshot(positions, velocities, net_forces, tags, imgs, Box(H, origin), dt)


def build_snapshot_methods(sampling_method):
    if sampling_method.requires_box_unwrapping:

        def positions(snapshot):
            L = np.diag(snapshot.box.H)
            return snapshot.positions[:, :3] + L * snapshot.images

    else:

        def positions(snapshot):
            return snapshot.positions

    @jit
    def indices(snapshot):
        return snapshot.ids

    @jit
    def momenta(snapshot):
        M = snapshot.vel_mass[:, 3:]
        V = snapshot.vel_mass[:, :3]
        return (M * V).flatten()

    @jit
    def masses(snapshot):
        return snapshot.vel_mass[:, 3:]

    return SnapshotMethods(jit(positions), indices, momenta, masses)


def build_helpers(context, sampling_method):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if is_on_gpu(context):
        cupy = importlib.import_module("cupy")
        view = cupy.asarray

        def sync_forces():
            cupy.cuda.get_current_stream().synchronize()

    else:
        utils = importlib.import_module(".utils", package="pysages.backends")
        view = utils.view

        def sync_forces():
            pass

    def bias(snapshot, state, sync_backend):
        """Adds the computed bias to the forces."""
        # TODO: check if this can be JIT compiled with numba.
        # Forces may be computed asynchronously on the GPU, so we need to
        # synchronize them before applying the bias.
        if state.bias is None:
            return
        sync_backend()
        forces = view(snapshot.forces)
        biases = view(state.bias.block_until_ready())
        forces[:, :3] += biases
        sync_forces()

    snapshot_methods = build_snapshot_methods(sampling_method)
    flags = sampling_method.snapshot_flags
    restore = partial(_restore, view)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), get_dimension(context))

    return helpers, restore, bias


def bind(
    wrapped_context: ContextWrapper,  sampling_method: SamplingMethod, callback: Callable, **kwargs
):
    context = wrapped_context.context
    wrapped_context.view = None
    wrapped_context.run = get_run_method(context)
    helpers, restore, bias = build_helpers(context, sampling_method)

    # take a simulation snapshot from the context
    with context:
        snapshot = take_snapshot(wrapped_context)

    method_bundle = sampling_method.build(snapshot, helpers)
    sync_and_bias = partial(bias, sync_backend=None)

    # create an instance of Sampler (which is DLExtSampler)
    sampler = Sampler(context, method_bundle, sync_and_bias, callback, restore)
    # and connect it with the LAMMPS object (context)
    set_post_force_hook(context, sampler)
    return sampler
