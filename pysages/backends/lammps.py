# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import importlib
import lammps

from functools import partial
from typing import Callable
from warnings import warn

from jax import jit, numpy as np
from jax.dlpack import from_dlpack as asarray
from lammps.dlext import (
    AccessLocation,
    AccessMode,
    HalfStepHook,
    SystemView,
    images,
    net_forces,
    positions_types,
    rtags,
    velocities_masses,
    DLextSampler
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


class Sampler(DLextSampler):
    def __init__(self, sysdef, method_bundle, bias, dt, callback: Callable):
        _ , initialize, update = method_bundle
        self.state = initialize()
        self.callback = callback
        self.bias = bias
        box = sysdef.getParticleData().getGlobalBox()
        self.pybox = self._get_pybox(box)
        self.dt = dt

        def python_update(positions, vel_mass, rtags, imgs, forces):
            positions = asarray(positions)
            vel_mass = asarray(vel_mass)
            ids = asarray(rtags)
            images = asarray(imgs)
            forces = asarray(forces)
            snap = Snapshot(positions=positions,
                            vel_mass = vel_mass,
                            forces=forces,
                            ids=ids,
                            images=images,
                            box=self.pybox,
                            dt=self.dt)
            self.state = update(snap, self.state)
            self.bias(snap, self.state)
            if self.callback:
                self.callback(snap, self.state, 0)

        super().__init__(sysdef, python_update)

    def _get_pybox(self, box):
        L = box.getL()
        xy = box.getTiltFactorXY()
        xz = box.getTiltFactorXZ()
        yz = box.getTiltFactorYZ()
        lo = box.getLo()
        H = (
            (L.x, xy * L.y, xz * L.z),
            (0.0,      L.y, yz * L.z),
            (0.0,      0.0,      L.z)
        )
        origin = (lo.x, lo.y, lo.z)
        return Box(H, origin)


if hasattr(AccessLocation, "OnDevice"):
    def default_location():
        return AccessLocation.OnDevice
else:
    def default_location():
        return AccessLocation.OnHost


def is_on_gpu(context):
    return context.on_gpu()


def take_snapshot(wrapped_context, location = default_location()):
    #
    context = wrapped_context.context
    sysview = wrapped_context.view
    #
    positions = asarray(positions_types(sysview, location, AccessMode.Read))
    vel_mass = asarray(velocities_masses(sysview, location, AccessMode.Read))
    forces = asarray(net_forces(sysview, location, AccessMode.ReadWrite))
    ids = asarray(rtags(sysview, location, AccessMode.Read))
    imgs = asarray(images(sysview, location, AccessMode.Read))
    #
    box = sysview.particle_data().getGlobalBox()
    L = box.getL()
    xy = box.getTiltFactorXY()
    xz = box.getTiltFactorXZ()
    yz = box.getTiltFactorYZ()
    lo = box.getLo()
    H = (
        (L.x, xy * L.y, xz * L.z),
        (0.0,      L.y, yz * L.z),
        (0.0,      0.0,      L.z)
    )
    origin = (lo.x, lo.y, lo.z)
    dt = context.integrator.dt
    #
    return Snapshot(positions, vel_mass, forces, ids, imgs, Box(H, origin), dt)


def update_snapshot(snapshot, sysview, location = default_location()):
    #
    positions = asarray(positions_types(sysview, location, AccessMode.Read))
    vel_mass = asarray(velocities_masses(sysview, location, AccessMode.Read))
    forces = asarray(net_forces(sysview, location, AccessMode.ReadWrite))
    ids = asarray(rtags(sysview, location, AccessMode.Read))
    imgs = asarray(images(sysview, location, AccessMode.Read))
    #
    return Snapshot(positions, vel_mass, forces, ids, imgs, snapshot.box, snapshot.dt)


def build_snapshot_methods(sampling_method):
    if sampling_method.requires_box_unwrapping:
        def positions(snapshot):
            L = np.diag(snapshot.box.H)
            return snapshot.positions[:, :3] + L * snapshot.images
    else:
        def positions(snapshot):
            return snapshot.positions

    def indices(snapshot):
        return snapshot.ids

    def momenta(snapshot):
        M = snapshot.vel_mass[:, 3:]
        V = snapshot.vel_mass[:, :3]
        return (M * V).flatten()

    def masses(snapshot):
        return snapshot.vel_mass[:, 3:]

    return SnapshotMethods(jit(positions), jit(indices), jit(momenta), jit(masses))


def build_helpers(context, sampling_method):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if is_on_gpu(context):
        cupy = importlib.import_module("cupy")
        view = cupy.asarray

        def sync_forces():
            cupy.cuda.get_current_stream().synchronize()
    else:
        utils = importlib.import_module(".utils", package = "pysages.backends")
        view = utils.view

        def sync_forces():
            pass

    def bias(snapshot, state, sync_backend):
        """Adds the computed bias to the forces."""
        # TODO: check if this can be JIT compiled with numba.
        # Forces may be computed asynchronously on the GPU, so we need to
        # synchronize them before applying the bias.
        sync_backend()
        forces = view(snapshot.forces)
        biases = view(state.bias.block_until_ready())
        forces[:, :3] += biases
        sync_forces()

    snapshot_methods = build_snapshot_methods(sampling_method)
    flags = sampling_method.snapshot_flags
    restore = partial(_restore, view)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), restore)

    return helpers, bias


def bind(
    wrapped_context: ContextWrapper,
    sampling_method: SamplingMethod,
    callback: Callable,
    **kwargs
):
    context = wrapped_context.context
    helpers, bias = build_helpers(context, sampling_method)

    sysview = SystemView(context.system_definition)
    wrapped_context.view = sysview
    wrapped_context.run = lammps.run

    snapshot = take_snapshot(wrapped_context)
    method_bundle = sampling_method.build(snapshot, helpers)
    sync_and_bias = partial(bias, sync_backend = sysview.synchronize)
    #
    sampler = Sampler(context.system_definition, method_bundle, sync_and_bias, context.integrator.dt, callback)
    context.integrator.cpp_integrator.setHalfStepHook(sampler)
    #
    CONTEXTS_SAMPLERS[context] = sampler
    #
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
