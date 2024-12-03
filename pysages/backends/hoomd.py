# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import importlib
from functools import partial
from warnings import warn

import hoomd
from hoomd import md
from hoomd.dlext import AccessLocation, AccessMode, DLExtSampler, SystemView
from jax import jit
from jax import numpy as np
from jax.dlpack import from_dlpack

from pysages.backends import snapshot as pbs
from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.typing import Callable, Optional
from pysages.utils import copy

SamplerBase = DLExtSampler

# TODO: Figure out a way to automatically tie the lifetime of Sampler
# objects to the contexts they bind to
CONTEXTS_SAMPLERS = {}


if getattr(hoomd, "__version__", "").startswith("2."):

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
    if not hasattr(hoomd.dlext, "__version__"):

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


if hasattr(AccessLocation, "OnDevice"):

    def default_location():
        return AccessLocation.OnDevice

else:

    def default_location():
        return AccessLocation.OnHost


class Sampler(SamplerBase):
    """
    HOOMD-blue HalfStepHook that connects PySAGES sampling methods to HOOMD-blue simulations.

    Parameters
    ----------

    context: hoomd.Simulation
        The HOOMD-blue simulation instance to which the PySAGES sampling machinery will be hooked.

    sampling_method: pysages.methods.SamplingMethod
        The sampling method used.

    callbacks: Optional[Callback]
        A callback. Some methods define one for logging, but it can also be user-defined.

    location: hoomd.dlext.AccessLocation
        Device where the simulation data will be retrieved.
    """

    def __init__(
        self, context, sampling_method, callback: Optional[Callable], location=default_location()
    ):
        self.context = context
        self.callback = callback
        self.location = location
        self.view = SystemView(get_system(context))

        self.box = Box(*get_global_box(self.view))
        self.dt = get_timestep(self.context)
        self.update_box = lambda: self.box  # NOTE: extend for NPT simulations

        super().__init__(self.view, self._update_callback, location, AccessMode.Read)

        # Create initial snapshot and setup sampling method
        snapshot = self.take_snapshot()  # sets `self.snapshot`
        helpers, restore, bias = build_helpers(context, sampling_method)
        _, initialize, method_update = sampling_method.build(snapshot, helpers)

        # Initialize state and store method references
        self.state = initialize()
        self._restore = restore
        self._method_update = method_update
        self._bias = bias

    def restore(self, prev_snapshot):
        """Restore the simulation state from a previous snapshot."""
        self.snapshot = prev_snapshot
        self.forward_data(self._restore_callback, self.location, AccessMode.Overwrite, 0)

    def take_snapshot(self):
        """Take a snapshot of the current simulation state."""
        self.forward_data(self._snapshot_callback, self.location, AccessMode.Read, 0)
        return self.snapshot

    def _pack_snapshot(self, positions, vel_mass, forces, rtags, images):
        return Snapshot(
            from_dlpack(positions),
            from_dlpack(vel_mass),
            from_dlpack(forces),
            from_dlpack(rtags),
            from_dlpack(images),
            self.update_box(),
            self.dt,
        )

    # NOTE: The order of the callbacks arguments do not match that of the `Snapshot` attributes
    def _restore_callback(self, positions, vel_mass, rtags, images, forces, _):
        snapshot = self._pack_snapshot(positions, vel_mass, forces, rtags, images)
        self._restore(snapshot, self.snapshot)

    def _snapshot_callback(self, positions, vel_mass, rtags, images, forces, _):
        snapshot = self._pack_snapshot(positions, vel_mass, forces, rtags, images)
        self.snapshot = copy(snapshot)

    def _update_callback(self, positions, vel_mass, rtags, images, forces, timestep):
        snapshot = self._pack_snapshot(positions, vel_mass, forces, rtags, images)
        self.state = self._method_update(snapshot, self.state)
        self._bias(snapshot, self.state, self.view.synchronize)
        if self.callback:
            self.callback(snapshot, self.state, timestep)


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
    utils = importlib.import_module(".utils", package="pysages.backends")

    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if is_on_gpu(context):
        sync_forces, view = utils.cupy_helpers()

    else:
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
    restore = partial(pbs.restore, view)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), dimensionality)

    return helpers, restore, bias


def get_global_box(system_view):
    """Get the box and origin of a HOOMD-blue simulation."""
    box = system_view.particle_data.getGlobalBox()
    L = box.getL()
    xy = box.getTiltFactorXY()
    xz = box.getTiltFactorXZ()
    yz = box.getTiltFactorYZ()
    lo = box.getLo()
    H = ((L.x, xy * L.y, xz * L.z), (0.0, L.y, yz * L.z), (0.0, 0.0, L.z))
    origin = (lo.x, lo.y, lo.z)
    return H, origin


def get_timestep(context):
    """Get the timestep of a HOOMD-blue simulation."""
    return get_integrator(context).dt


def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    context = sampling_context.context
    sampling_method = sampling_context.method

    sampler = Sampler(context, sampling_method, callback)
    set_half_step_hook(context, sampler)
    CONTEXTS_SAMPLERS[context] = sampler

    sampling_context.run = get_run_method(context)

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
