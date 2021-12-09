# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from functools import partial
from typing import Callable

from jax import jit, numpy as np
from jax.dlpack import from_dlpack as asarray
from jax.lax import cond
from openmm_dlext import ContextView, DeviceType, Force

from pysages.backends.core import ContextWrapper
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
    restore as _restore,
    restore_vm as _restore_vm,
)
from pysages.methods import SamplingMethod
from pysages.utils import try_import

import importlib
import jax
import openmm_dlext as dlext

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")


class Sampler:
    def __init__(self, method_bundle, bias, callback: Callable):
        snapshot, initialize, update = method_bundle
        self.snapshot = snapshot
        self.state = initialize()
        self._update = update
        self.bias = bias
        self.callback = callback

    def update(self, timestep=0):
        self.state = self._update(self.snapshot, self.state)
        self.bias(self.snapshot, self.state)
        if self.callback:
            self.callback(self.snapshot, self.state, timestep)


def is_on_gpu(view: ContextView):
    return view.device_type() == DeviceType.GPU


def take_snapshot(wrapped_context):
    #
    context = wrapped_context.context.context  # extra indirection for OpenMM
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
    # OpenMM doesn't have images
    return Snapshot(positions, vel_mass, forces, ids, None, Box(H, origin), dt)


def identity(x):
    return x


def safe_divide(v, invm):
    return cond(invm[0] == 0, lambda x: v, lambda x: np.divide(v, x), invm)


def build_snapshot_methods(context, sampling_method):
    if is_on_gpu(context):
        def unpack(vel_mass):
            return vel_mass[:, :3], vel_mass[:, 3:]

        def indices(snapshot):
            return snapshot.ids.argsort()

        def masses(snapshot):
            return snapshot.vel_mass[:, 3:]
    else:
        unpack = identity

        def indices(snapshot):
            return snapshot.ids

        def masses(snapshot):
            _, invM = snapshot.vel_mass
            return jax.vmap(safe_divide)(1, invM)

    def positions(snapshot):
        return snapshot.positions

    def momenta(snapshot):
        V, invM = unpack(snapshot.vel_mass)
        return jax.vmap(safe_divide)(V, invM).flatten()

    return SnapshotMethods(jit(positions), jit(indices), jit(momenta), jit(masses))


def build_helpers(context, sampling_method):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if is_on_gpu(context):
        cupy = importlib.import_module("cupy")
        view = cupy.asarray

        restore_vm = _restore_vm

        def sync_forces():
            cupy.cuda.get_current_stream().synchronize()

        @jit
        def adapt(biases):
            return np.int64(2**32 * biases.T)

    else:
        utils = importlib.import_module(".utils", package = "pysages.backends")
        view = utils.view

        adapt = identity

        def sync_forces():
            pass

        def restore_vm(view, snapshot, prev_snapshot):
            # TODO: Check if we can omit modifying the masses
            # (in general the masses are unlikely to change)
            velocities = view(snapshot.vel_mass[0])
            masses = view(snapshot.masses[1])
            velocities[:] = view(prev_snapshot.vel_mass[0])
            masses[:] = view(prev_snapshot.vel_mass[1])

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

    snapshot_methods = build_snapshot_methods(context, sampling_method)
    flags = sampling_method.snapshot_flags
    restore = partial(_restore, view, restore_vm = restore_vm)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), restore)

    return helpers, bias


def check_integrator(context):
    integrator = context.getIntegrator()
    if (
        isinstance(integrator, openmm.VariableLangevinIntegrator) or
        isinstance(integrator, openmm.VariableVerletIntegrator)
    ):
        raise ValueError("Variable step size integrators are not supported")


def bind(
    wrapped_context: ContextWrapper,
    sampling_method: SamplingMethod,
    callback: Callable,
    **kwargs
):
    # For OpenMM we need to store a Simulation object as the context,
    simulation = wrapped_context.context
    context = simulation.context
    check_integrator(context)
    force = Force()
    force.add_to(context)  # OpenMM will handle the lifetime of the force
    wrapped_context.view = force.view(context)
    wrapped_context.run = simulation.step
    helpers, bias = build_helpers(wrapped_context.view, sampling_method)
    snapshot = take_snapshot(wrapped_context)
    method_bundle = sampling_method.build(snapshot, helpers)
    sync_and_bias = partial(bias, sync_backend = wrapped_context.view.synchronize)
    sampler = Sampler(method_bundle, sync_and_bias, callback)
    force.set_callback_in(context, sampler.update)
    return sampler
