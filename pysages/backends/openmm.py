# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES


import importlib
import jax
import jax.numpy as jnp
import openmm_dlext as dlext
import pysages.backends.common as common
import simtk.openmm as openmm
import simtk.unit as unit

from typing import Callable
from functools import partial
from jax.dlpack import from_dlpack as asarray
from jax.lax import cond
from jaxlib.xla_extension import DeviceArray as JaxArray
from openmm_dlext import ContextView, DeviceType, Force
from pysages.backends.common import HelperMethods
from pysages.backends.snapshot import Box, Snapshot

from .core import ContextWrapper


class ContextWrapperOpenMM(ContextWrapper):
    def __init__(self, context, force):
        super().__init__(context)
        self.view = force.view(context)
        self.synchronize = self.view.synchronize


class Sampler:
    def __init__(self, method_bundle, bias, callback: Callable):
        snapshot, initialize, update = method_bundle
        self.snapshot = snapshot
        self.state = initialize()
        self._update = update
        self.bias = bias
        self.callback = callback
    #
    def update(self, timestep=0):
        self.state = self._update(self.snapshot, self.state)
        self.bias(self.snapshot, self.state)
        if self.callback:
            self.callback(self.snapshot, self.state, timestep)


def is_on_gpu(view: ContextView):
    return view.device_type() == DeviceType.GPU


def take_snapshot(wrapped_context):
    #
    context = wrapped_context.context
    context_view = wrapped_context.view
    #
    positions = asarray(dlext.positions(context_view))
    forces = asarray(dlext.forces(context_view))
    ids = asarray(dlext.atom_ids(context_view))
    #
    velocities = asarray(dlext.velocities(context_view))
    if is_on_gpu(context_view):
        vel_mass = velocities
    else:
        inverse_masses = asarray(dlext.inverse_masses(context_view))
        vel_mass = (velocities, inverse_masses.reshape((-1, 1)))
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
        #
        restore_vm = common.restore_vm
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
        def restore_vm(view, snapshot, prev_snapshot):
            # TODO: Check if we can omit modifying the masses
            # (in general the masses are unlikely to change)
            velocities = view(snapshot.vel_mass[0])
            masses = view(snapshot.masses[1])
            velocities[:] = view(prev_snapshot.vel_mass[0])
            masses[:] = view(prev_snapshot.vel_mass[1])
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
    restore = partial(common.restore, view, restore_vm = restore_vm)
    #
    return HelperMethods(jax.jit(indices), jax.jit(momenta), restore), bias


def check_integrator(context):
    integrator = context.getIntegrator()
    if (
        isinstance(integrator, openmm.VariableLangevinIntegrator) or
        isinstance(integrator, openmm.VariableVerletIntegrator)
    ):
        raise ValueError("Variable step size integrators are not supported")


def bind(context, sampling_method, callback: Callable, force = Force(), **kwargs):
    check_integrator(context)
    #
    force.add_to(context)
    wrapped_context = ContextWrapperOpenMM(context, force)
    helpers, bias = build_helpers(wrapped_context.view)
    snapshot = take_snapshot(wrapped_context)
    method_bundle = sampling_method(snapshot, helpers)
    sync_and_bias = partial(bias, sync_backend = wrapped_context.synchronize)
    #
    sampler = Sampler(method_bundle, sync_and_bias, callback)
    force.set_callback_in(context, sampler.update)
    #
    return sampler
