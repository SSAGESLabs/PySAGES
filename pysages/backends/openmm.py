# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md


import importlib
import jax

import jax.numpy as jnp
import openmm_dlext as dlext
import simtk.openmm as openmm
import simtk.unit as unit

from jax.lax import cond
from functools import partial
from openmm_dlext import ContextView, DeviceType, Force
from pysages.backends.snapshot import Box, Snapshot

from jax.dlpack import from_dlpack as asarray
from jaxlib.xla_extension import DeviceArray as JaxArray

from .core import ContextWrapper

class ContextWrapperOpenMM(ContextWrapper):
    def __init__(self, context, force):
        super().__init__(context)
        self.view = force.view(context)
        self.synchronize = self.view.synchronize



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
        #numba = importlib.import_module("numba")
        #view = cuda.as_cuda_array
        #
        def sync_forces():
            cupy.cuda.get_current_stream().synchronize()
        #
        #apply_args_types = [
        #    "(float32[:, :], int64[:, :])",
        #    "(float64[:, :], int64[:, :])"
        #]
        # In OpenMM, forces are transposed and always stored in
        # 64-bit fixed-point buffers on the CUDA Platform. See
        # http://docs.openmm.org/latest/developerguide/developer.html#accumulating-forces
        #@numba.guvectorize(apply_args_types, "(n, m) -> (m, n)", target = "cuda")
        #def apply(biases, forces):
        #    m, n = forces.shape
        #    for i in range(m):
        #        for j in range(n):
        #            forces[i, j] += 2**32 * biases[j, i]
        #
        @jax.jit
        def adapt(biases):
            return jnp.int64(2**32 * biases.T)
        #
        def unpack(vel_mass):
            return vel_mass[:, :3], vel_mass[:, 3:]
        #
        def indices(ids):
            return ids.argsort()
    else:
        utils = importlib.import_module(".utils", package = "pysages.backends")
        view = utils.view
        #
        def apply(biases, forces):
            forces += biases
        #
        def sync_forces(): pass
        #
        def identity(x):
            return x
        #
        unpack = indices = adapt = identity
    #
    @jax.vmap
    def safe_divide(v, invm):
        return cond(
            invm[0] == 0, lambda x: v, lambda x: jnp.divide(v, x), invm
        )
    #
    def momenta(vel_mass):
        V, invM = unpack(vel_mass)
        return safe_divide(V, invM).flatten()
    #
    def bias(snapshot, state, sync_backend):
        """Adds the computed bias to the forces."""
        # TODO: check if this can be JIT compiled with numba.
        biases = adapt(state.bias)
        # Forces may be computed asynchronously on the GPU, so we need to
        # synchronize them before applying the bias.
        sync_backend()
        forces = view(snapshot.forces)
        biases = view(biases.block_until_ready())
        forces += biases
        sync_forces()
    #
    return jax.jit(indices), jax.jit(momenta), bias


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
    indices, momenta, bias = build_helpers(wrapped_context.view)
    snapshot = take_snapshot(wrapped_context)
    method_bundle = sampling_method(snapshot, (indices, momenta))
    sync_and_bias = partial(bias, sync_backend = wrapped_context.synchronize)
    #
    sampler = Sampler(method_bundle, sync_and_bias)
    force.set_callback_in(context, sampler.update)
    #
    return sampler
