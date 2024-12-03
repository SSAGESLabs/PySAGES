# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from inspect import Parameter, signature

from ase.calculators.calculator import Calculator
from jax import jit
from jax import numpy as np

from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.backends.utils import view
from pysages.typing import Callable
from pysages.utils import ToCPU, copy


class Sampler(Calculator):
    """
    Thin wrapper around the calculator of the `ase.atoms.Atoms` object of an
    `ase.md.MolecularDynamics` instance. The `get_forces` method will return
    the wrapped calculator forces plus the biasing forces from the sampling
    method being used.
    """

    def __init__(self, context, method_bundle, callback: Callable):
        initial_snapshot, initialize, method_update = method_bundle

        atoms = context.atoms
        self.implemented_properties = atoms.calc.implemented_properties
        ps = set(self.implemented_properties).intersection(("energy", "forces"))
        err = "Calculator does not support 'energy' or 'forces' calculations"
        assert len(ps) == 2, err

        self.atoms = atoms
        self.callback = callback
        self.snapshot = initial_snapshot
        self.state = initialize()
        self.update = method_update

        sig = signature(atoms.calc.calculate).parameters
        self._calculator = atoms.calc
        self._context = context
        self._biased_forces = initial_snapshot.forces
        self._default_properties = list(_calculator_defaults(sig, "properties"))
        self._default_changes = list(_calculator_defaults(sig, "system_changes"))
        for p in ("energy", "forces"):
            if p not in self._default_properties:
                self._default_properties.append(p)
        self._get_forces = atoms.calc.get_forces
        self._md_step = context.step

        # Swap the original step method to add the bias
        context.step = lambda: self._md_step(self.biased_forces)
        # Swap the atoms calculator with this wrapper
        atoms.calc = self

    def __getattr__(self, name):
        return getattr(self._calculator, name)

    @property
    def biased_forces(self):
        return view(copy(self._biased_forces, ToCPU()))

    def calculate(self, atoms=None, **kwargs):
        properties = kwargs.get("properties", self._default_properties)
        system_changes = kwargs.get("system_changes", self._default_changes)
        self._calculator.calculate(atoms, properties, system_changes)

    def get_forces(self, atoms=None):
        forces = self._get_forces(atoms)
        self.snapshot = take_snapshot(self._context, forces)
        self.state = self.update(self.snapshot, self.state)
        new_forces = self.snapshot.forces
        if self.state.bias is not None:
            new_forces += self.state.bias
        if self.callback:
            timestep = self._context.get_number_of_steps()
            self.callback(self.snapshot, self.state, timestep)
        self._biased_forces = new_forces
        return self.biased_forces

    def restore(self, prev_snapshot):
        atoms = self.atoms
        momenta, masses = prev_snapshot.vel_mass
        atoms.set_positions(prev_snapshot.positions)
        atoms.set_masses(masses.flatten())  # masses need to be set before momenta
        atoms.set_momenta(momenta, apply_constraint=False)
        atoms.set_cell(list(prev_snapshot.box.H))
        self.snapshot = prev_snapshot

    def take_snapshot(self):
        return copy(self.snapshot)


def take_snapshot(simulation, forces=None):
    atoms = simulation.atoms

    positions = np.asarray(atoms.get_positions())
    forces = np.asarray(atoms.get_forces(md=True) if forces is None else forces)
    ids = np.arange(len(positions))
    momenta = np.asarray(atoms.get_momenta())
    masses = np.asarray(atoms.get_masses()).reshape(-1, 1)
    vel_mass = (momenta, masses)

    H = (*atoms.cell,)
    origin = (0.0, 0.0, 0.0)
    dt = simulation.dt

    # ASE doesn't use images explicitely
    return Snapshot(positions, vel_mass, forces, ids, None, Box(H, origin), dt)


def _calculator_defaults(sig, arg, default=[]):
    fallback = Parameter("_", Parameter.KEYWORD_ONLY, default=default)
    val = sig.get(arg, fallback).default
    return val if type(val) is list else default


def build_snapshot_methods(context, sampling_method):
    def indices(snapshot):
        return snapshot.ids

    def masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    def positions(snapshot):
        return snapshot.positions

    def momenta(snapshot):
        P, _ = snapshot.vel_mass
        return P.flatten()

    return SnapshotMethods(jit(positions), jit(indices), jit(momenta), jit(masses))


def build_helpers(context, sampling_method):
    def dimensionality():
        return 3  # are all ASE simulations boxes 3-dimensional?

    snapshot_methods = build_snapshot_methods(context, sampling_method)
    flags = sampling_method.snapshot_flags
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), dimensionality)

    return helpers


def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    """
    Entry point for the backend code, it gets called when the simulation
    context is wrapped within `pysages.run`.
    """
    context = sampling_context.context
    sampling_method = sampling_context.method
    snapshot = take_snapshot(context)
    helpers = build_helpers(sampling_context, sampling_method)
    method_bundle = sampling_method.build(snapshot, helpers)
    sampler = Sampler(context, method_bundle, callback)
    sampling_context.run = context.run
    return sampler
