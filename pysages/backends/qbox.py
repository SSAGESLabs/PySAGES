# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
This module defines the Sampler class, which enables any PySAGES SamplingMethod to be
hooked to a Qbox simulation instance.
"""

from jax import jit
from jax import numpy as np
from plum import Val, type_parameter

from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.typing import Callable, Optional
from pysages.utils import dispatch, last, parse_array


class Sampler:
    """
    Allows performing enhanced sampling simulations with Qbox as a backend.

    Parameters
    ----------

    context: QboxContext
        Contains a running instance of a Qbox simulation to which the PySAGES sampling
        machinery will be hooked.

    sampling_method: SamplingMethod
        The sampling method to be used.

    callbacks: Optional[Callback]
        Some methods define callbacks for logging, but it can also be user-defined.
    """

    def __init__(self, context, sampling_method, callback: Optional[Callable]):
        self.context = context
        self.callback = callback

        self.snapshot = self.take_snapshot()
        helpers, bias, atom_names, cv_indices = build_helpers(context, sampling_method)
        _, initialize, method_update = sampling_method.build(self.snapshot, helpers)

        # Initialize external forces for each atom
        for i in cv_indices:
            name = atom_names[i]
            # Initialize with zero force
            cmd = f"extforce define atomic {name} {name} 0.0 0.0 0.0"
            context.process_input(cmd)

        self.state = initialize()
        self._update_box = lambda: self.snapshot.box
        self._method_update = method_update
        self._bias = bias

    def _pack_snapshot(self, masses, ids, box, dt):
        """Returns the dynamic properties of the system."""
        positions = atom_property(self.context, "position")
        velocities = atom_property(self.context, "velocity")
        forces = atom_property(self.context, "force")
        return Snapshot(positions, (velocities, masses), forces, ids, None, box, dt)

    def _update_snapshot(self):
        """Updates the snapshot with the latest properties from Qbox."""
        snapshot = self.snapshot
        _, masses = snapshot.vel_mass
        return self._pack_snapshot(masses, snapshot.ids, self._update_box(), snapshot.dt)

    def restore(self, prev_snapshot):
        """Replaces this sampler's snapshot with `prev_snapshot`."""
        context = self.context
        names = atom_property(context, "name")
        positions = prev_snapshot.positions
        velocities, _ = prev_snapshot.vel_mass

        for name, x, v in zip(names, positions, velocities):
            cmd = f"move {name} to {x[0]} {x[1]} {x[2]} {v[0]} {v[1]} {v[2]}"
            context.process_input(cmd)

        # Recompute ground-state energies and forces.
        # NOTE: Check in the future how to use Qbox `load` and `save` commands to also
        # include the electronic wave function data.
        context.process_input(f"run 0 {context.nitscf} {context.nite}")
        self.snapshot = self._update_snapshot()

    def take_snapshot(self):
        """Returns a copy of the current snapshot of the system."""
        masses = atom_property(self.context, "mass")
        ids = np.arange(len(masses))
        snapshot_box = Box(*box(self.context))
        dt = timestep(self.context)
        return self._pack_snapshot(masses, ids, snapshot_box, dt)

    def update(self, timestep):
        """Update the sampling method state and apply bias."""
        self.snapshot = self._update_snapshot()
        self.state = self._method_update(self.snapshot, self.state)
        self._bias(self.snapshot, self.state)
        if self.callback:
            self.callback(self.snapshot, self.state, timestep)

    def run(self, nsteps: int):
        """Run the Qbox simulation for nsteps."""
        cmd = f"run {self.context.niter} {self.context.nitscf} {self.context.nite}"
        for step in range(nsteps):
            # Send run command to Qbox for a single step
            self.context.process_input(cmd)
            # Update sampling method state after each step
            self.update(step)


def build_snapshot_methods(sampling_method):
    """
    Builds methods for retrieving snapshot properties in a format useful for collective
    variable calculations.
    """

    def positions(snapshot):
        return snapshot.positions

    def indices(snapshot):
        return snapshot.ids

    def momenta(snapshot):
        V, M = snapshot.vel_mass
        return (M * V).flatten()

    def masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    return SnapshotMethods(positions, indices, jit(momenta), masses)


def build_helpers(context, sampling_method):
    """
    Builds helper methods used for restoring snapshots and biasing a simulation.
    """
    # Precompute atom names since they won't change
    atom_names = atom_property(context, "name")

    cv_indices = set()
    for cv in sampling_method.cvs:
        cv_indices.update(n.item() for n in cv.indices)

    def extforce_cmd(name, force):
        return f"extforce set {name} {force[0]} {force[1]} {force[2]}"

    def bias(snapshot, state):
        """Adds the computed bias to the forces using Qbox's extforce command."""
        if state.bias is None:
            return
        # Generate and send all extforce commands
        context.process_input(extforce_cmd(atom_names[i], state.bias[i]) for i in cv_indices)

    snapshot_methods = build_snapshot_methods(sampling_method)
    flags = sampling_method.snapshot_flags
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), lambda: 3)

    return helpers, bias, atom_names, cv_indices


@dispatch
def atom_property(context, prop: str):
    return atom_property(context, *property_handler(context, Val(prop)))


@dispatch
def atom_property(context, xml_tag, extract, gather):
    atomset = last(context.state.iter("atomset"))
    if atomset is None:
        context.process_input("run 0")
        atomset = last(context.state.iter("atomset"))
    return gather(extract(elem) for elem in atomset.iter(xml_tag))


@dispatch
def property_handler(context, prop: Val["name"]):  # noqa: F821
    return (
        "atom",  # xml_tag
        (lambda s: s.attrib["name"]),  # extract
        list,  # gather
    )


@dispatch
def property_handler(context, prop: Val["mass"]):  # noqa: F821
    return (
        "atom",  # xml_tag
        (lambda s: context.species_masses[s.attrib["species"]]),  # extract
        (lambda d: np.array(list(d)).reshape(-1, 1)),  # gather
    )


@dispatch
def property_handler(context, prop: Val):
    return (
        type_parameter(prop),  # xml_tag
        (lambda s: s.text),  # extract
        (lambda d: parse_array(" ".join(d))),  # gather
    )


def box(context):
    elem = last(context.state.iter("unit_cell"))
    if elem is None:
        context.process_input("print cell")
        elem = context.state.find("unit_cell")
    cell_vecs = " ".join(elem.attrib.values())
    H = parse_array(cell_vecs, transpose=True)
    origin = np.array([0.0, 0.0, 0.0])
    return Box(H, origin)


def timestep(context):
    context.process_input("print dt")
    elem = context.state.find("cmd")
    return float(elem.tail.strip("\ndt= "))


def bind(sampling_context: SamplingContext, callback: Optional[Callable], **kwargs):
    """
    Sets up and returns a Sampler which enables performing enhanced sampling simulations.

    This function takes a `sampling_context` that has its context attribute as an instance
    of a `QboxContext,` and creates a `Sampler` object that connects the PySAGES
    sampling method to the Qbox simulation. It also modifies the `sampling_context`'s
    `view` and `run` attributes to call the Qbox `run` command.
    """
    context = sampling_context.context
    sampler = Sampler(context, sampling_context.method, callback)
    sampling_context.run = sampler.run

    return sampler
