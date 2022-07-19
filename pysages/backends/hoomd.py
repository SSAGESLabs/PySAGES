# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import importlib
import hoomd

from functools import partial
from typing import Callable
from warnings import warn

from jax import jit, numpy as np
from jax.dlpack import from_dlpack as asarray
from hoomd.dlext import (
    AccessLocation,
    AccessMode,
    SystemView,
    images,
    net_forces,
    positions_types,
    rtags,
    velocities_masses,
    DLExtSampler,
)

from pysages.backends.core import ContextWrapper
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
    restore as _restore,
)
from pysages.methods import SamplingMethod


# TODO: Figure out a way to automatically tie the lifetime of Sampler
# objects to the contexts they bind to
CONTEXTS_SAMPLERS = {}


class Sampler(DLExtSampler):
    def __init__(self, sysview, method_bundle, bias, callback: Callable, restore):
        initial_snapshot, initialize, method_update = method_bundle
        self.state = initialize()
        self.callback = callback
        self.bias = bias
        self.box = initial_snapshot.box
        self.dt = initial_snapshot.dt
        self._restore = restore

        def update(positions, vel_mass, rtags, images, forces, timestep):
            snapshot = Snapshot(
                asarray(positions),
                asarray(vel_mass),
                asarray(forces),
                asarray(rtags),
                asarray(images),
                self.box,
                self.dt,
            )
            self.state = method_update(snapshot, self.state)
            self.bias(snapshot, self.state)
            if self.callback:
                self.callback(snapshot, self.state, timestep)

        super().__init__(sysview, update, default_location(), AccessMode.Read)

    def restore(self, prev_snapshot):
        def restore_callback(positions, vel_mass, rtags, images, forces, n):
            snapshot = Snapshot(
                asarray(positions),
                asarray(vel_mass),
                asarray(forces),
                asarray(rtags),
                asarray(images),
                self.box,
                self.dt,
            )
            self._restore(snapshot, prev_snapshot)

        self.forward_data(restore_callback, default_location(), AccessMode.Overwrite)


if hasattr(AccessLocation, "OnDevice"):

    def default_location():
        return AccessLocation.OnDevice

else:

    def default_location():
        return AccessLocation.OnHost


def is_on_gpu(context):
    return context.on_gpu()


def take_snapshot(wrapped_context, location=default_location()):
    context = wrapped_context.context
    sysview = wrapped_context.view
    positions = asarray(positions_types(sysview, location, AccessMode.Read))
    vel_mass = asarray(velocities_masses(sysview, location, AccessMode.Read))
    forces = asarray(net_forces(sysview, location, AccessMode.ReadWrite))
    ids = asarray(rtags(sysview, location, AccessMode.Read))
    imgs = asarray(images(sysview, location, AccessMode.Read))

    box = sysview.particle_data().getGlobalBox()
    L = box.getL()
    xy = box.getTiltFactorXY()
    xz = box.getTiltFactorXZ()
    yz = box.getTiltFactorYZ()
    lo = box.getLo()
    H = ((L.x, xy * L.y, xz * L.z), (0.0, L.y, yz * L.z), (0.0, 0.0, L.z))
    origin = (lo.x, lo.y, lo.z)
    dt = context.integrator.dt

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
        if state.bias is not None:
            sync_backend()
            forces = view(snapshot.forces)
            biases = view(state.bias.block_until_ready())
            forces[:, :3] += biases
            sync_forces()

    snapshot_methods = build_snapshot_methods(sampling_method)
    flags = sampling_method.snapshot_flags
    restore = partial(_restore, view)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags))

    return helpers, restore, bias


def bind(
    wrapped_context: ContextWrapper, sampling_method: SamplingMethod, callback: Callable, **kwargs
):
    context = wrapped_context.context
    helpers, restore, bias = build_helpers(context, sampling_method)

    with SystemView(context.system_definition) as sysview:
        wrapped_context.view = sysview
        wrapped_context.run = hoomd.run

        snapshot = take_snapshot(wrapped_context)
        method_bundle = sampling_method.build(snapshot, helpers)
        sync_and_bias = partial(bias, sync_backend=sysview.synchronize)

        sampler = Sampler(sysview, method_bundle, sync_and_bias, callback, restore)
        context.integrator.cpp_integrator.setHalfStepHook(sampler)

        CONTEXTS_SAMPLERS[context] = sampler

    return sampler


def detach(context):
    """
    If pysages was bound to this context, this removes the corresponding
    `Sampler` object.
    """
    if context in CONTEXTS_SAMPLERS:
        context.integrator.cpp_integrator.removeHalfStepHook()
        del CONTEXTS_SAMPLERS[context]
    else:
        warn("This context has no sampler bound to it.")
