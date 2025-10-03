# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import importlib
from functools import partial

import jax
import openmm_dlext as dlext
from jax import jit
from jax import numpy as np
from jax.dlpack import from_dlpack
from jax.lax import cond
from openmm_dlext import ContextView, DeviceType, Force

from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.backends.snapshot import restore as _restore
from pysages.backends.snapshot import restore_vm as _restore_vm
from pysages.typing import Callable
from pysages.utils import check_device_array, copy, try_import

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")


class Sampler:
    def __init__(self, method_bundle, context, bias, callback: Callable, restore):
        initial_snapshot, initialize, method_update = method_bundle
        self.state = initialize()
        self.bias = bias
        self.callback = callback
        self.snapshot = initial_snapshot
        self._restore = restore
        self._update = method_update
        self._update_box = lambda: get_box(context)

    def update(self, timestep=0):
        self.snapshot = self.update_snapshot()
        self.state = self._update(self.snapshot, self.state)
        self.bias(self.snapshot, self.state)
        if self.callback:
            self.callback(self.snapshot, self.state, timestep)

    def restore(self, prev_snapshot):
        self._restore(self.snapshot, prev_snapshot)

    def take_snapshot(self):
        return copy(self.snapshot)

    def update_snapshot(self):
        box = self._update_box()
        return self.snapshot._replace(box=box)


def is_on_gpu(view: ContextView):
    return view.device_type() == DeviceType.GPU


def take_snapshot(sampling_context):
    context = sampling_context.context.context  # extra indirection for OpenMM
    context_view = sampling_context.view

    positions = from_dlpack(dlext.positions(context_view))
    forces = from_dlpack(dlext.forces(context_view))
    ids = from_dlpack(dlext.atom_ids(context_view))

    velocities = from_dlpack(dlext.velocities(context_view))
    if is_on_gpu(context_view):
        vel_mass = velocities
    else:
        inverse_masses = from_dlpack(dlext.inverse_masses(context_view))
        vel_mass = (velocities, inverse_masses.reshape((-1, 1)))

    check_device_array(positions)  # currently, we only support `DeviceArray`s

    box = get_box(context)
    dt = context.getIntegrator().getStepSize() / unit.picosecond

    # OpenMM doesn't have images
    return Snapshot(positions, vel_mass, forces, ids, None, box, dt)


def get_box(context):
    box_vectors = context.getSystem().getDefaultPeriodicBoxVectors()
    a = box_vectors[0].value_in_unit(unit.nanometer)
    b = box_vectors[1].value_in_unit(unit.nanometer)
    c = box_vectors[2].value_in_unit(unit.nanometer)
    H = ((a[0], b[0], c[0]), (a[1], b[1], c[1]), (a[2], b[2], c[2]))
    origin = (0.0, 0.0, 0.0)
    return Box(H, origin)


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
    utils = importlib.import_module(".utils", package="pysages.backends")

    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if is_on_gpu(context):
        restore_vm = _restore_vm
        sync_forces, view = utils.cupy_helpers()

        @jit
        def adapt(biases):
            return np.int64(2**32 * biases.T)

    else:
        adapt = identity
        view = utils.view

        def restore_vm(view, snapshot, prev_snapshot):
            # TODO: Check if we can omit modifying the masses
            # (in general the masses are unlikely to change)
            velocities = view(snapshot.vel_mass[0])
            masses = view(snapshot.vel_mass[1])
            velocities[:] = view(prev_snapshot.vel_mass[0])
            masses[:] = view(prev_snapshot.vel_mass[1])

        def sync_forces():
            pass

    def bias(snapshot, state, sync_backend):
        """Adds the computed bias to the forces."""
        if state.bias is None:
            return
        # Forces may be computed asynchronously on the GPU, so we need to
        # synchronize them before applying the bias.
        sync_backend()
        biases = adapt(state.bias)
        forces = view(snapshot.forces)
        forces += view(biases.block_until_ready())
        sync_forces()

    def dimensionality():
        return 3  # all OpenMM simulations boxes are 3-dimensional

    snapshot_methods = build_snapshot_methods(context, sampling_method)
    flags = sampling_method.snapshot_flags
    restore = partial(_restore, view, restore_vm=restore_vm)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), dimensionality)

    return helpers, restore, bias


def check_integrator(context):
    integrator = context.getIntegrator()
    if isinstance(integrator, openmm.VariableLangevinIntegrator) or isinstance(
        integrator, openmm.VariableVerletIntegrator
    ):
        raise ValueError("Variable step size integrators are not supported")


def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    # For OpenMM we need to store a Simulation object as the context,
    simulation = sampling_context.context
    context = simulation.context
    check_integrator(context)
    sampling_method = sampling_context.method
    force = Force()
    force.add_to(context)  # OpenMM will handle the lifetime of the force
    sampling_context.view = force.view(context)
    sampling_context.run = simulation.step
    helpers, restore, bias = build_helpers(sampling_context.view, sampling_method)
    snapshot = take_snapshot(sampling_context)
    method_bundle = sampling_method.build(snapshot, helpers)
    sync_and_bias = partial(bias, sync_backend=sampling_context.view.synchronize)
    sampler = Sampler(method_bundle, context, sync_and_bias, callback, restore)
    force.set_callback_in(context, sampler.update)
    return sampler
