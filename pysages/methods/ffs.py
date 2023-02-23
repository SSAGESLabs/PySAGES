# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Forward Flux Sampling (FFS).

Implementation of the direct version of the FFS algorithm.
FFS uses a series of non-intersecting interfaces between the initial and the final states.
The initial and final states are defined in terms of an order parameter.
The method allows to calculate rate constants and generate transition paths.
"""

import sys
from typing import Callable, NamedTuple, Optional
from warnings import warn

from jax import numpy as np

from pysages.backends import SamplingContext
from pysages.methods.core import SamplingMethod, generalize
from pysages.utils import JaxArray, dispatch


class FFSState(NamedTuple):
    xi: JaxArray
    bias: Optional[JaxArray]

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class FFS(SamplingMethod):
    """
    Constructor of the Forward Flux Sampling method.

    Parameters
    ----------
    self : FFS
        See parent class
    cvs:
        See parent class
    args:
        See parent class
    kwargs:
        See parent class

    Attributes
    ----------
    snapshot_flags:
        Indicate the particle properties required from a snapshot.
    """

    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, **kwargs):
        kwargs["cv_grad"] = False
        super().__init__(cvs, **kwargs)

    def build(self, snapshot, helpers):
        self.helpers = helpers
        return _ffs(self, snapshot, helpers)


# We override the default run method as FFS is algorithmically fairly different
@dispatch
def run(
    method: FFS,
    context_generator: Callable,
    timesteps: int,
    dt: float,
    win_i: float,
    win_l: float,
    Nw: int,
    sampling_steps_basin: int,
    Nmax_replicas: int,
    verbose: bool = False,
    callback: Optional[Callable] = None,
    context_args: Optional[dict] = None,
    **kwargs,
):
    """
    Direct version of the Forward Flux Sampling algorithm.
    [Phys. Rev. Lett. 94, 018104 (2005)](https://doi.org/10.1103/PhysRevLett.94.018104)
    [J. Chem. Phys. 124, 024102 (2006)](https://doi.org/10.1063/1.2140273)

    Arguments
    ---------
    method: FFS

    context_generator: Callable
        User defined function that sets up a simulation context with the backend.
        Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
        and `simtk.openmm.Simulation` for OpenMM. The function gets `context_args`
        unpacked for additional user arguments.

    timesteps: int
        Number of timesteps the simulation is running.

    dt: float
        Timestep of the simulation

    win_i: float
        Initial window for the system

    win_l: float
        Last window to be calculated in ffs

    Nw: int
        Number of equally spaced windows

    sampling_steps_basin: int
        Period for sampling configurations in the basin

    Nmax_replicas: int
        Number of stored configuration for each window

    verbose: bool
        If True more information will be logged (useful for debbuging).

    callback: Optional[Callable] = None
        Allows for user defined actions into the simulation workflow of the method.
        `kwargs` gets passed to the backend `run` function.

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.

    NOTE:
        The current implementation runs a single simulation/replica,
        but multiple concurrent simulations can be scripted on top of this.
    """

    context_args = {} if context_args is None else context_args

    context = context_generator(**context_args)
    context_args["context"] = context
    sampling_context = SamplingContext(context, method, callback)

    with sampling_context:
        sampler = sampling_context.sampler
        xi = sampler.state.xi.block_until_ready()
        windows = np.linspace(win_i, win_l, num=Nw)

        is_configuration_good = check_input(windows, xi, verbose=verbose)
        if not is_configuration_good:
            raise ValueError("Bad initial configuration")

        run = sampling_context.run
        helpers = method.helpers
        cv = method.cv

        reference_snapshot = sampler.take_snapshot()

        # We Initially sample from basin A
        # TODO: bundle the arguments into data structures
        ini_snapshots = basin_sampling(
            Nmax_replicas,
            sampling_steps_basin,
            windows,
            run,
            sampler,
            reference_snapshot,
            helpers,
            cv,
        )

        # Calculate initial flow
        phi_a, snaps_0 = initial_flow(
            Nmax_replicas, dt, windows, ini_snapshots, run, sampler, helpers, cv
        )

        write_to_file(phi_a)
        hist = np.zeros(len(windows))
        hist = hist.at[0].set(phi_a)

        # Calculate conditional probability for each window
        for k in range(1, len(windows)):
            if k == 1:
                old_snaps = snaps_0
            prob, w1_snapshots = running_window(windows, k, old_snaps, run, sampler, helpers, cv)
            write_to_file(prob)
            hist = hist.at[k].set(prob)
            old_snaps = increase_snaps(w1_snapshots, snaps_0)
            print(f"size of snapshots: {len(old_snaps)}\n")

        K_t = np.prod(hist)
        write_to_file("# Flux Constant")
        write_to_file(K_t)

    return sampling_context.sampler.state


def _ffs(method, snapshot, helpers):
    """
    Internal function that generates an `initialize` and an `update` functions.
    `initialize` is ran once just before the time integration starts and `update`
    is called after each simulation timestep.

    Arguments
    ---------
    method: FFS

    snapshot:
        PySAGES snapshot of the simulation (backend dependent).

    helpers
        Helper function bundle as generated by
        `SamplingMethod.context.get_backend().build_helpers`.

    Returns
    -------
    Tuple `(snapshot, initialize, update)` as described above.
    """
    cv = method.cv

    # initialize method
    def initialize():
        xi = cv(helpers.query(snapshot))
        return FFSState(xi, None)

    def update(state, data):
        xi = cv(data)
        return FFSState(xi, None)

    return snapshot, initialize, generalize(update, helpers)


def write_to_file(value):
    with open("ffs_results.dat", "a+") as f:
        f.write(str(value) + "\n")


# Since snapshots are depleted each window, this function restores the list to
# its initial values. This only works with stochastic integrators like BD or
# Langevin, for other, velocity resampling is needed
def increase_snaps(windows, initial_w):
    if len(windows) > 0:
        ratio = len(initial_w) // len(windows)
        windows = windows * ratio

    return windows


def check_input(grid, xi, verbose=False):
    """
    Verify whether the initial configuration is a good one.
    """
    is_good = xi < grid[0]

    if is_good:
        print("Good initial configuration\n")
        print(xi)
    elif verbose:
        print(xi)

    return is_good


def basin_sampling(
    max_num_snapshots, sampling_time, grid, run, sampler, reference_snapshot, helpers, cv
):
    """
    Sampling of basing configurations for initial flux calculations.
    """
    basin_snapshots = []
    win_A = grid[0]
    xi = sampler.state.xi.block_until_ready()

    print("Starting basin sampling\n")
    while len(basin_snapshots) < int(max_num_snapshots):
        run(sampling_time)
        xi = sampler.state.xi.block_until_ready()

        if np.all(xi < win_A):
            snap = sampler.take_snapshot()
            basin_snapshots.append(snap)
            print("Storing basing configuration with cv value:\n")
            print(xi)
        else:
            sampler.restore(reference_snapshot)
            xi = cv(helpers.query(reference_snapshot))
            print("Restoring basing configuration since system left basin with cv value:\n")
            print(xi)

    print(f"Finish sampling basin with {max_num_snapshots} snapshots\n")

    return basin_snapshots


def initial_flow(Num_window0, timestep, grid, initial_snapshots, run, sampler, helpers, cv):
    """
    Selects snapshots from list generated with `basin_sampling`.
    """

    success = 0
    time_count = 0.0
    window0_snaps = []
    win_A = grid[0]

    for i in range(0, Num_window0):
        print(f"Initial stored configuration: {i}\n")
        snap = initial_snapshots[i]
        sampler.restore(snap)
        xi = cv(helpers.query(snap))
        print(xi)

        has_reached_A = False
        while not has_reached_A:
            # TODO: make the number of timesteps below a parameter of the method.
            run(1)
            time_count += timestep
            xi = sampler.state.xi.block_until_ready()

            if np.all(xi >= win_A) and np.all(xi < grid[1]):
                success += 1
                has_reached_A = True

                if len(window0_snaps) <= Num_window0:
                    snap = sampler.take_snapshot()
                    window0_snaps.append(snap)

                break

    print(f"Finish Initial Flow with {success} succeses over {time_count} time\n")
    phi_a = float(success) / (time_count)

    return phi_a, window0_snaps


def running_window(grid, step, old_snapshots, run, sampler, helpers, cv):
    success = 0
    new_snapshots = []
    win_A = grid[0]
    win_value = grid[int(step)]
    has_conf_stored = False

    for i in range(0, len(old_snapshots)):
        snap = old_snapshots[i]
        sampler.restore(snap)
        xi = cv(helpers.query(snap))
        print(f"Stored configuration: {i} of window: {step}\n")
        print(xi)

        # limit running time to avoid zombie trajectories
        # this can be probably be improved
        running = True
        while running:
            run(1)
            xi = sampler.state.xi.block_until_ready()

            if np.all(xi < win_A):
                running = False
            elif np.all(xi >= win_value):
                snap = sampler.take_snapshot()
                new_snapshots.append(snap)
                success += 1
                running = False
                if not has_conf_stored:
                    has_conf_stored = True

    if success == 0:
        warn(f"Unable to estimate probability, exiting early at window {step}\n")
        sys.exit(0)

    if len(new_snapshots) > 0:
        prob_local = float(success) / len(old_snapshots)
        print(f"Finish window {step} with {len(new_snapshots)} snapshots\n")
        return prob_local, new_snapshots
