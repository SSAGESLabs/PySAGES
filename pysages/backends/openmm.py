# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md


import importlib
import jax

import openmm_dlext as dlext
import simtk.openmm as openmm
import simtk.unit as unit

from functools import partial
from openmm_dlext import ContextView, DeviceType, Force
from pysages.backends.snapshot import Box, Snapshot

from jax.dlpack import from_dlpack as asarray
from jaxlib.xla_extension import DeviceArray as JaxArray


Int64 = jax.numpy.int64


class ContextWrapper:
    def __init__(self, context, force):
        self.view = force.view(context)
        self.context = context
        self.synchronize = self.view.synchronize


class Sampler:
    def __init__(self, method_bundle, bias):
        snapshot, initialize, update = method_bundle
        self.snapshot = snapshot
        self.state = initialize()
        self.update_from = update
        self.bias = bias
    #
    def update(self):
        self.state = self.update_from(self.snapshot, self.state)
        self.bias(self.snapshot, self.state)


def is_on_gpu(view: ContextView):
    return view.device_type() == DeviceType.GPU


def choose_backend(context):
    if is_on_gpu(context):
        return jax.lib.xla_bridge.get_backend("gpu")
    return jax.lib.xla_bridge.get_backend("cpu")


def take_snapshot(wrapped_context):
    #
    context = wrapped_context.context
    context_view = wrapped_context.view
    backend = choose_backend(context_view)
    #
    positions = asarray(dlext.positions(context_view), backend)
    forces = asarray(dlext.forces(context_view), backend)
    ids = asarray(dlext.atom_ids(context_view), backend)
    #
    if is_on_gpu(context_view):
        vel_mass = asarray(dlext.velocities(context_view), backend)
    else:
        inverse_masses = asarray(dlext.inverse_masses(context_view), backend)
        vel_mass = (vel_mass, inverse_masses)
    #
    box_vectors = context.getSystem().getDefaultPeriodicBoxVectors()
    a = box_vectors[0].value_in_unit(unit.nanometer)
    b = box_vectors[1].value_in_unit(unit.nanometer)
    c = box_vectors[2].value_in_unit(unit.nanometer)
    H = (
        (a[0], b[0], c[0]),
        (a[1], b[1], c[1]),
        (a[2], b[2], c[2])
    )
    origin = (0.0, 0.0, 0.0)
    dt = context.getIntegrator().getStepSize() / unit.picosecond
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
        #
        @jax.jit
        def adapt(biases: JaxArray):
            return view(Int64(2**32 * biases.T))
        #
        def unpack(vel_mass):
            return vel_mass, vel_mass[:, 3:]
        #
        def indices(ids):
            return ids.argsort()
    else:
        utils = importlib.import_module(".utils", package = "pysages.backends")
        view = utils.view
        adapt = view
        #
        def sync_forces(): pass
        #
        def unpack(vel_mass):
            return vel_mass
        #
        def indices(ids):
            return ids
    #
    def bias(snapshot, state, sync_backend):
        """Adds the computed bias to the forces."""
        # TODO: Factor out the views so we can eliminate two function calls here.
        # Also, check if this can be JIT compiled with numba.
        biases = adapt(state.bias.block_until_ready())
        # Forces may be computed asynchronously on the GPU, so we need to
        # synchronize them before applying the bias.
        sync_backend()
        forces = view(snapshot.forces)
        forces += biases
        sync_forces()
    #
    def momenta(vel_mass):
        V, IM = unpack(vel_mass)
        return jax.numpy.divide(V, IM).flatten()
    #
    return bias, jax.jit(indices), jax.jit(momenta)


def check_integrator(context):
    integrator = context.getIntegrator()
    if (
        isinstance(integrator, openmm.VariableLangevinIntegrator) or
        isinstance(integrator, openmm.VariableVerletIntegrator)
    ):
        raise ValueError("Variable step size integrators are not supported")


def bind(context, sampling_method, force = Force(), **kwargs):
    check_integrator(context)
    #
    force.add_to(context)
    wrapped_context = ContextWrapper(context, force)
    bias, indices, momenta = build_helpers(wrapped_context.view)
    snapshot = take_snapshot(wrapped_context)
    method_bundle = sampling_method(snapshot, (indices, momenta))
    sync_and_bias = partial(bias, sync_backend = wrapped_context.synchronize)
    #
    sampler = Sampler(method_bundle, sync_and_bias)
    force.set_callback_in(context, sampler.update)
    #
    return sampler
