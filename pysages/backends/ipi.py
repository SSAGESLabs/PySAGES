# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
This module defines the Sampler class, which is an i-PI force field that enables any PySAGES
SamplingMethod to be hooked to an i-PI simulation instance.
"""

from pathlib import Path
from weakref import finalize

import numpy
from ipi.engine.forcefields import FFEval, FFPlumed
from ipi.utils.depend import dstrip
from ipi.utils.messages import verbosity
from ipi.utils.scripting import InteractiveSimulation
from jax import jit
from jax import numpy as np

# Import PySAGES components
from pysages.backends import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.typing import Callable
from pysages.utils import identity


# %%
class Sampler(FFEval):
    """
    i-PI force field that connects PySAGES sampling methods to i-PI simulations.
    This class inherits from FFEval and follows the PySAGES backend API pattern
    while replicating the FFPlumed interface for compatibility.

    This class provides a clean interface for replacing FFPlumed instances with
    PySAGES-based force fields, post-i-PI initialization.
    """

    # TODO: Replace FFPlumed instances with PySAGES-based force fields, during i-PI initialization.
    # TODO: Add PySAGES-based force fields to the simulation, during or after i-PI initialization.

    def __init__(self, context, sampling_method, callback=None, **kwargs):
        """
        Initialize the PySAGES context for an existing InteractiveSimulation.

        Parameters
        ----------
        context: InteractiveSimulation
            The existing i-PI simulation object
        sampling_method: pysages.methods.SamplingMethod
            The PySAGES sampling method to use
        callback: Optional[Callable]
            Optional callback for logging or custom operations
        **kwargs
            Additional arguments passed to FFPySAGES
        """
        name = kwargs.get("name", "pysages")

        super().__init__(
            name=name, dopbc=kwargs.get("dopbc", False), threaded=kwargs.get("threaded", False)
        )

        context.fflist[name] = self
        context.bind()

        self.context = context
        if launch_drivers := kwargs.get("launch_drivers"):
            self.context.drivers = launch_drivers()

        initial_snapshot = self.take_snapshot(extras={})
        helpers = build_helpers(sampling_method)
        _, initialize, method_update = sampling_method.build(initial_snapshot, helpers)

        # PySAGES integration
        self.callback = callback
        self.snapshot = initial_snapshot._replace(extras=context.get_structures())
        self.state = initialize()
        self._method_update = method_update

        if context.syslist[0].motion.enstype in {"npt", "nst", "scnpt"}:
            self.update_box = identity(lambda r: Box(r["cell"][0], np.zeros(3)))
        else:
            self.update_box = identity(lambda r: self.snapshot.box)

        self.extras = {"raw": ""}
        self.extras_tags = kwargs.get("extras_tags", [])

        # FFPlumed compatibility attributes
        self.init_file = kwargs.get("init_file", "")
        self.compute_work = True
        self.plumed_dat = ""
        self.plumed_data = {}
        self.plumed_extras = []
        self.plumed_step = 0
        self.smotion_step = 0

        # We won't compute these for now
        self.bias_potential = numpy.zeros(1, float)
        self.virial = numpy.zeros((3, 3), float)

    def evaluate(self, request):
        """
        Evaluate forces and potential energy for the current configuration.
        This is called by i-PI during force evaluation and replicates the FFPlumed interface.

        Parameters
        ----------
        request : dict
            Dictionary containing positions and cell information from i-PI.
            Results will be written to request["result"].
        """
        # Update snapshot for PySAGES, keep extras empty to avoid JAX processing issues
        snapshot = self.update_snapshot(request)
        self.state = self._method_update(snapshot, self.state)
        self.snapshot = snapshot._replace(extras=self.context.get_structures())

        # Call callback if provided
        if self.callback is not None:
            self.callback(self.snapshot, self.state)

        xi = numpy.asarray(self.state.xi).flatten()
        biasing_forces = numpy.asarray(self.state.bias).flatten()

        values = (*xi, self.bias_potential)
        for p, x in zip(self.extras_tags, values):
            self.extras[p] = x

        # [potential, forces, virial, extra]
        request["result"] = [
            self.bias_potential,
            biasing_forces,
            self.virial,
            self.extras,
        ]
        request["status"] = "Done"

    def mtd_update(self, pos, cell):
        """
        Update metadynamics bias (replicates FFPlumed.mtd_update interface).
        This method is called by i-PI's metadynamics motion class.

        Parameters
        ----------
        pos : array
            Current positions
        cell : array
            Current cell

        Returns
        -------
        float
            Work done by the bias (0.0 for now)
        """
        # Update step counter
        self.smotion_step += 1

        bias_before = self.bias_potential  # TODO: Use `bias_potential(self.state)` instead
        # snapshot = self.take_snapshot(extras={})
        # state = self._method_update(snapshot, self.state)
        bias_after = self.bias_potential  # TODO: Use `bias_potential(state)` instead

        work = (bias_before - bias_after).item()

        return work

    def queue(self, atoms, cell, reqid=-1):
        return super().queue(atoms, cell, reqid, template={"momenta": dstrip(atoms.p)})

    def restore(self, prev_snapshot):
        """Restore the internal state of the replaced force fields."""
        self.context.set_structures(self._snapshot_extras)
        self.snapshot = prev_snapshot

    def take_snapshot(self, extras=None):
        """
        Get a snapshot of the current system state.

        Returns
        -------
        Snapshot
            A PySAGES snapshot of the system state
        """
        system = self.context.syslist[0]

        positions = ipi_to_jax(system.beads.qc).reshape(-1, 3)
        # PySAGES can work with momenta directly - no need for (vel, mass) tuple
        momenta = ipi_to_jax(system.beads.pc).reshape(-1, 3)
        ids = np.arange(len(positions))
        box = Box(ipi_to_jax(system.cell.h), np.zeros(3))
        dt = system.motion.dt
        extras = self.context.get_structures() if extras is None else extras

        return Snapshot(positions, momenta, None, ids, None, box, dt, extras)

    def update_snapshot(self, request, extras={}):
        """
        Update the snapshot with the current system state.
        """
        positions = np.asarray(request["pos"]).reshape(-1, 3)
        momenta = np.asarray(request["momenta"]).reshape(-1, 3)
        box = self.update_box(request)
        return self.snapshot._replace(positions=positions, vel_mass=momenta, box=box, extras=extras)


# %%
def ipi_to_jax(x):
    return np.asarray(dstrip(x))


# %%
def build_snapshot_methods(sampling_method):
    """
    Build methods for retrieving snapshot properties in a format useful for collective
    variable calculations.

    Parameters
    ----------
    sampling_method : callable
        The sampling method to use

    Returns
    -------
    SnapshotMethods
        Methods for retrieving snapshot properties
    """

    def indices(snapshot):
        return snapshot.ids

    def masses(snapshot):
        return None

    def positions(snapshot):
        return snapshot.positions

    def momenta(snapshot):
        # vel_mass contains momenta directly
        return snapshot.vel_mass.flatten()

    return SnapshotMethods(jit(positions), jit(indices), jit(momenta), jit(masses))


# %%
def build_helpers(sampling_method):
    """
    Build helper methods used for restoring snapshots and biasing a simulation.

    Parameters
    ----------
    sampling_method : callable
        The sampling method to use

    Returns
    -------
    HelperMethods
        Helper methods for snapshot operations
    """

    def dimensionality():
        return 3  # i-PI simulations are always 3D

    snapshot_methods = build_snapshot_methods(sampling_method)
    querier = build_data_querier(snapshot_methods, sampling_method.snapshot_flags)
    helpers = HelperMethods(querier, dimensionality)

    return helpers


# %%
def clone(sim):
    system = sim.syslist[0]
    assert len(sim.syslist) == 1, "Only single-system simulations supported"
    assert len(system.ensemble.bcomp) == 1, "Only centroid-biased systems supported"
    assert (
        sum(isinstance(ff, FFPlumed) for ff in sim.fflist.values()) == 1
    ), "Only simulations with a single biasing force supported"

    def unlink_sockets(addresses):
        for a in addresses:
            Path(a).unlink(missing_ok=True)

    context = InteractiveSimulation('<simulation verbosity="quiet"></simulation>')
    context.__dict__.update(sim.chk.status.fetch().__dict__)

    addresses = [
        ff.socket.sockets_prefix + ff.socket.address
        for ff in context.fflist.values()
        if hasattr(ff, "socket")
    ]
    finalize(context, unlink_sockets, addresses)
    unlink_sockets(addresses)

    return context


# %%
def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    """
    Binds a sampling method to an i-PI simulation context for force field switching.
    This function serves as the entry point for creating Samplers from PySAGES.

    Parameters
    ----------
    sampling_context: SamplingContext
        The sampling context containing the simulation and method
        (context attribute refers to InteractiveSimulation)
    callback: Optional[Callable]
        Optional callback for logging or custom operations

    Returns
    -------
    Sampler
        Context object for managing PySAGES force field switching
    """
    orig_context = sampling_context.context
    context = clone(orig_context)

    name, ff = next((k, v) for k, v in context.fflist.items() if isinstance(v, FFPlumed))
    kwargs["name"] = name
    kwargs["init_file"] = ff.init_file
    kwargs["extras_tags"] = [str(s) for s in ff.plumed_extras]

    # Create a Sampler for force field switching.
    sampler = Sampler(context, sampling_context.method, callback=callback, **kwargs)

    context.chk.status = orig_context.chk.status
    verbosity.level = orig_context.chk.status.verbosity.fetch()
    sampling_context.context = context
    sampling_context.run = context.run

    return sampler
