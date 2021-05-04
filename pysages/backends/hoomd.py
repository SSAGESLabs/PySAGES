# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md


import importlib
import jax

from functools import partial
from hoomd.dlext import (
    AccessLocation, AccessMode, HalfStepHook, SystemView,
    net_forces, positions_types, rtags, velocities_masses,
)
from pysages.backends.snapshot import Box, Snapshot

from jax.dlpack import from_dlpack as asarray


class ContextWrapper:
    def __init__(self, context):
        self.sysview = SystemView(context.system_definition)
        self.context = context
        self.synchronize = self.sysview.synchronize


class Sampler(HalfStepHook):
    def __init__(self, method_bundle, bias):
        super().__init__()
        #
        snapshot, initialize, update = method_bundle
        self.snapshot = snapshot
        self.state = initialize()
        self._update = update
        self.bias = bias
    #
    def update(self, timestep):
        self.state = self._update(self.snapshot, self.state)
        self.bias(self.snapshot, self.state)


if hasattr(AccessLocation, "OnDevice"):
    def default_location():
        return AccessLocation.OnDevice
else:
    def default_location():
        return AccessLocation.OnHost


def is_on_gpu(context):
    return context.on_gpu()


def choose_backend(context, requested_location):
    if is_on_gpu(context) and requested_location != AccessLocation.OnHost:
        return jax.lib.xla_bridge.get_backend("gpu")
    return jax.lib.xla_bridge.get_backend("cpu")


def take_snapshot(wrapped_context, location = default_location()):
    #
    context = wrapped_context.context
    sysview = wrapped_context.sysview
    backend = choose_backend(context, location)
    #
    positions = asarray(positions_types(sysview, location, AccessMode.Read), backend)
    vel_mass = asarray(velocities_masses(sysview, location, AccessMode.Read), backend)
    forces = asarray(net_forces(sysview, location, AccessMode.ReadWrite), backend)
    ids = asarray(rtags(sysview, location, AccessMode.Read), backend)
    #
    box = sysview.particle_data().getGlobalBox()
    L  = box.getL()
    xy = box.getTiltFactorXY()
    xz = box.getTiltFactorXZ()
    yz = box.getTiltFactorYZ()
    lo = box.getLo()
    H = (
        (L.x, xy * L.y, xz * L.z, 0.0),  # Last column is a hack until
        (0.0,      L.y, yz * L.z, 0.0),  # https://github.com/google/jax/issues/4196
        (0.0,      0.0,      L.z, 0.0)   # gets fixed
    )
    origin = (lo.x, lo.y, lo.z)
    dt = context.integrator.dt
    #
    return Snapshot(positions, vel_mass, forces, ids, Box(H, origin), dt)


def build_helpers(context):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if is_on_gpu(context):
        cupy = importlib.import_module("cupy")
        view = cupy.asarray
        #
        def sync_forces():
            cupy.cuda.get_current_stream().synchronize()
    else:
        utils = importlib.import_module(".utils", package = "pysages.backends")
        view = utils.view
        #
        def sync_forces(): pass
    #
    def bias(snapshot, state, sync_backend):
        """Adds the computed bias to the forces."""
        # TODO: check if this can be JIT compiled with numba.
        biases = view(state.bias.block_until_ready())
        # Forces may be computed asynchronously on the GPU, so we need to
        # synchronize them before applying the bias.
        sync_backend()
        forces = view(snapshot.forces)
        forces += biases
        sync_forces()
    #
    def indices(ids):
        return ids
    #
    def momenta(vel_mass):
        M = vel_mass[:, 3:]
        V = vel_mass
        return jax.numpy.multiply(M, V).flatten()
    #
    return bias, jax.jit(indices), jax.jit(momenta)


def bind(context, sampling_method, **kwargs):
    #
    bias, indices, momenta = build_helpers(context)
    #
    wrapped_context = ContextWrapper(context)
    snapshot = take_snapshot(wrapped_context)
    method_bundle = sampling_method(snapshot, (indices, momenta))
    sync_and_bias = partial(bias, sync_backend = wrapped_context.synchronize)
    #
    sampler = Sampler(method_bundle, sync_and_bias)
    context.integrator.cpp_integrator.setHalfStepHook(sampler)
    #
    return sampler
