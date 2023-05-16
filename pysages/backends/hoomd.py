# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import importlib
from functools import partial
from typing import Callable
from warnings import warn

import hoomd
from hoomd import md
from hoomd.dlext import (
    AccessLocation,
    AccessMode,
    DLExtSampler,
    SystemView,
    images,
    net_forces,
    positions_types,
    rtags,
    velocities_masses,
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

# TODO: Figure out a way to automatically tie the lifetime of Sampler
# objects to the contexts they bind to
CONTEXTS_SAMPLERS = {}


if getattr(hoomd, "__version__", "").startswith("2."):
    SamplerBase = DLExtSampler

    def is_on_gpu(context):
        return context.on_gpu()

    def get_integrator(context):
        return context.integrator

    def get_run_method(context):
        return hoomd.run

    def get_system(context):
        return context.system

    def set_half_step_hook(context, half_step_hook):
        context.integrator.cpp_integrator.setHalfStepHook(half_step_hook)

    def remove_half_step_hook(context):
        context.integrator.cpp_integrator.removeHalfStepHook()

else:

    class SamplerBase(DLExtSampler, md.HalfStepHook):
        def __init__(self, sysview, update, location, mode):
            md.HalfStepHook.__init__(self)
            DLExtSampler.__init__(self, sysview, update, location, mode)

    def is_on_gpu(context):
        return not isinstance(context.device, hoomd.device.CPU)

    def get_integrator(context):
        return context.operations.integrator

    def get_run_method(context):
        context.run(0)  # ensure that the context is properly initialized
        return context.run

    def get_system(context):
        return context._cpp_sys

    def set_half_step_hook(context, half_step_hook):
        context.operations.integrator.half_step_hook = half_step_hook

    def remove_half_step_hook(context):
        context.operations.integrator.half_step_hook = None


class Sampler(SamplerBase):
    def __init__(self, sysview, method_bundle, bias, callback: Callable, restore):
        initial_snapshot, initialize, method_update = method_bundle

        def update(positions, vel_mass, rtags, images, forces, timestep):
            snapshot = self._pack_snapshot(positions, vel_mass, forces, rtags, images)
            self.state = method_update(snapshot, self.state)
            self.bias(snapshot, self.state)
            if self.callback:
                self.callback(snapshot, self.state, timestep)

        super().__init__(sysview, update, default_location(), AccessMode.Read)
        self.state = initialize()
        self.bias = bias
        self.box = initial_snapshot.box
        self.callback = callback
        self.dt = initial_snapshot.dt
        self._restore = restore

    def restore(self, prev_snapshot):
        def restore_callback(positions, vel_mass, rtags, images, forces, n):
            snapshot = self._pack_snapshot(positions, vel_mass, forces, rtags, images)
            self._restore(snapshot, prev_snapshot)

        self.forward_data(restore_callback, default_location(), AccessMode.Overwrite, 0)

    def take_snapshot(self):
        container = []

        def snapshot_callback(positions, vel_mass, rtags, images, forces, n):
            snapshot = self._pack_snapshot(positions, vel_mass, forces, rtags, images)
            container.append(copy(snapshot))

        self.forward_data(snapshot_callback, default_location(), AccessMode.Read, 0)
        return container[0]

    def _pack_snapshot(self, positions, vel_mass, forces, rtags, images):
        return Snapshot(
            asarray(positions),
            asarray(vel_mass),
            asarray(forces),
            asarray(rtags),
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


def take_snapshot(sampling_context, location=default_location()):
    context = sampling_context.context
    sysview = sampling_context.view
    positions = copy(asarray(positions_types(sysview, location, AccessMode.Read)))
    vel_mass = copy(asarray(velocities_masses(sysview, location, AccessMode.Read)))
    forces = copy(asarray(net_forces(sysview, location, AccessMode.ReadWrite)))
    ids = copy(asarray(rtags(sysview, location, AccessMode.Read)))
    imgs = copy(asarray(images(sysview, location, AccessMode.Read)))

    check_device_array(positions)  # currently, we only support `DeviceArray`s

    box = sysview.particle_data.getGlobalBox()
    L = box.getL()
    xy = box.getTiltFactorXY()
    xz = box.getTiltFactorXZ()
    yz = box.getTiltFactorYZ()
    lo = box.getLo()
    H = ((L.x, xy * L.y, xz * L.z), (0.0, L.y, yz * L.z), (0.0, 0.0, L.z))
    origin = (lo.x, lo.y, lo.z)
    dt = get_integrator(context).dt

    return Snapshot(positions, vel_mass, forces, ids, imgs, Box(H, origin), dt)


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

    def dimensionality():
        return 3  # all HOOMD-blue simulations boxes are 3-dimensional

    snapshot_methods = build_snapshot_methods(sampling_method)
    flags = sampling_method.snapshot_flags
    restore = partial(_restore, view)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), dimensionality)

    return helpers, restore, bias


def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    context = sampling_context.context
    sampling_method = sampling_context.method
    sysview = SystemView(get_system(context))
    sampling_context.view = sysview
    sampling_context.run = get_run_method(context)
    helpers, restore, bias = build_helpers(context, sampling_method)

    with sysview:
        snapshot = take_snapshot(sampling_context)

    method_bundle = sampling_method.build(snapshot, helpers)
    sync_and_bias = partial(bias, sync_backend=sysview.synchronize)
    sampler = Sampler(sysview, method_bundle, sync_and_bias, callback, restore)
    set_half_step_hook(context, sampler)

    CONTEXTS_SAMPLERS[context] = sampler

    return sampler


def detach(context):
    """
    If pysages was bound to this context, this removes the corresponding
    `Sampler` object.
    """
    if context in CONTEXTS_SAMPLERS:
        remove_half_step_hook(context)
        del CONTEXTS_SAMPLERS[context]
    else:
        warn("This context has no sampler bound to it.")
