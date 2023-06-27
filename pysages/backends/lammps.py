# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# Maintainer: ndtrung

import importlib
from functools import partial
from typing import Callable

from lammps import dlext
from lammps.dlext import (
    ExecutionSpace,
    FixDLExt,
    LAMMPSView,
)
from jax import jit
from jax import numpy as np
from jax.dlpack import from_dlpack as asarray

from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.backends.snapshot import restore as _restore
from pysages.utils import check_device_array, copy


def get_dimension(context):
    return context.extract_setting("dimension")


def get_timestep(context):
    return context.extract_global("dt")


def get_global_box(context):
    boxlo, boxhi, xy, yz, xz, *_ = context.extract_box()
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
class Sampler(FixDLExt):
    def __init__(self, context, method_bundle, bias, callback: Callable, restore):
        super().__init__(context, "1 all dlext space device".split())

        initial_snapshot, initialize, method_update = method_bundle

        def update(timestep):
            self.state = method_update(self.snapshot, self.state)
            self.bias(self.snapshot, self.state)
            if self.callback:
                self.callback(self.snapshot, self.state, timestep)

        self.set_callback(update)
        self.state = initialize()
        self.bias = bias
        self.callback = callback
        self.snapshot = initial_snapshot
        self.dt = initial_snapshot.dt
        self.box = initial_snapshot.box
        self._restore = restore

    def restore(self, prev_snapshot):
        self._restore(self.snapshot, prev_snapshot)

    def take_snapshot(self):
        location = default_location()

        positions = copy(asarray(dlext.positions(self.view, location)))
        types = copy(asarray(dlext.types(self.view, location)))
        velocities = copy(asarray(dlext.velocities(self.view, location)))
        masses = copy(asarray(dlext.masses(self.view, location)))
        forces = copy(asarray(dlext.forces(self.view, location)))
        tags = copy(asarray(dlext.tags(self.view, location)))
        imgs = copy(asarray(dlext.images(self.view, location)))

        vel_mass = (velocities, masses[types])

        return Snapshot(positions, vel_mass, forces, tags, imgs, self.box, self.dt)


if hasattr(ExecutionSpace, "kOnDevice"):

    def default_location():
        return ExecutionSpace.kOnDevice

else:

    def default_location():
        return ExecutionSpace.kOnHost


# build a Snapshot object that contains all the tensors from the context
def take_snapshot(sampling_context, location=default_location()):
    context = sampling_context.context
    view = sampling_context.view

    positions = copy(asarray(dlext.positions(view, location)))
    types = copy(asarray(dlext.types(view, location)))
    velocities = copy(asarray(dlext.velocities(view, location)))
    masses = copy(asarray(dlext.masses(view, location)))
    net_forces = copy(asarray(dlext.forces(view, location)))
    tags = copy(asarray(dlext.tags(view, location)))
    imgs = copy(asarray(dlext.images(view, location)))

    vel_mass = (velocities, masses[types])

    check_device_array(positions)  # currently, we only support `DeviceArray`s

    H, origin = get_global_box(context)
    dt = get_timestep(context)

    return Snapshot(positions, vel_mass, net_forces, tags, imgs, Box(H, origin), dt)


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
        V, M = snapshot.vel_mass
        return (M * V).flatten()

    @jit
    def masses(snapshot):
        return snapshot.vel_mass[:, 3:]

    return SnapshotMethods(jit(positions), indices, momenta, masses)


def build_helpers(sampling_context):
    context = sampling_context.context
    sampling_method = sampling_context.method
    view = sampling_context.view

    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if view.has_kokkos_cuda_enabled():
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


def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    context = sampling_context.context
    sampling_method = sampling_context.method
    view = LAMMPSView(context)
    sampling_context.view = view
    sampling_context.run = lambda n, **kwargs: context.command(f"run {n}")
    helpers, restore, bias = build_helpers(sampling_context)

    snapshot = take_snapshot(sampling_context)

    # the build() member function of a specific sampling method
    # (e.g. see abf.py) returns a triplet:
    #   initial_snapshot (object)  : initial configuration of the simulation context
    #   initialize       (function): returns the initial state of the sampling method
    #   method_update    (function): generalize() update a state depending on JIT or not
    method_bundle = sampling_method.build(snapshot, helpers)
    sync_and_bias = partial(bias, sync_backend=view.synchronize)
    return Sampler(view, method_bundle, sync_and_bias, callback, restore)
