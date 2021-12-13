# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES
"""
Adaptive Biasing Force (ABF) sampling method.

The implementation of the adaptive biasing force method here closely follows
https://doi.org/10.1063/1.2829861. One important difference is that the time
derivative of the product W·p (equation 9 of reference) is approximated by as
second order backward finite difference in the simulation timestep.
"""
# pylint: disable=invalid-name

from typing import NamedTuple

from jax import numpy as np, scipy

from pysages.grids import build_indexer
from pysages.methods.core import GriddedSamplingMethod, generalize
from pysages.utils import JaxArray


class ABFState(NamedTuple):
    """
    ABF internal state.

    Parameters
    ----------
    bias : JaxArray (Nparticles, 3)
        Array with biasing forces for each particle.
    xi : JaxArray (CV shape)
        Last collective variable recorded in the simulation.
    hist: JaxArray (grid.shape)
        Histogram of visits to the bins in the collective variable grid.
    force_sum: JaxArray (grid.shape, CV shape)
        Cumulative forces at each bin in the CV grid.
    force: JaxArray (grid.shape, CV shape)
        Average force at each bin of the CV grid.
    Wp: JaxArray (CV shape)
        Product of W matrix and momenta matrix for the current step.
    Wp_: JaxArray (CV shape)
        Product of W matrix and momenta matrix for the previous step.
    """
    bias: JaxArray
    xi: JaxArray
    hist: JaxArray
    force_sum: JaxArray
    force: JaxArray
    Wp: JaxArray
    Wp_: JaxArray

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class ABF(GriddedSamplingMethod):
    """
    Constructor of the ABF method.

    Parameters
    ----------
    self : ABF
        See parent class
    cvs:
        See parent class
    args:
        See parent class
    kwargs:
        An optional keyword argument `N: int` can be specified as
        threshold parameter before accounting for the full average
        of the adaptive biasing force  (defaults to 200).

    Attributes
    ----------
    snapshot_flags
      Indicating the fields required in a snapshot.
    """
    snapshot_flags = {"positions", "indices", "momenta"}

    def __init__(self, cvs, grid, *args, **kwargs):
        super().__init__(cvs, grid, *args, **kwargs)
        self.N = np.asarray(self.kwargs.get('N', 200))

    def build(self, snapshot, helpers, *args, **kwargs):
        """
        Build the functions for the execution of ABF

        Arguments
        ---------
        snapshot:
            PySAGES snapshot of the simulation (backend depend)
        helpers:
            Helper function bundle as generated by
            `SamplingMethod.context.get_backend().build_helpers`.

        Returns
        -------
        Tuple `(snapshot, initialize, update)` to run ABF simulations.
        """
        return _abf(self, snapshot, helpers)


def _abf(method, snapshot, helpers):
    """
    Internal function, that generates the init and update functions.

    Arguments
    ---------
    method: ABF
        class that generates the functions
    snapshot:
        PySAGES snapshot of the simulation (backend depend)
    helpers
        Helper function bundle as generated by i.e.
        `SamplingMethod.context.get_backend().build_helpers`

    Returns
    -------
    Tuple `(snapshot, initialize, update)` to run ABF simulations.
    """
    cv = method.cv
    grid = method.grid
    N = method.N

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    get_grid_index = build_indexer(grid)

    def initialize():
        """
        Internal function that generates the first ABFstate with correctly shaped JaxArrays

        Returns
        -------
        ABFState
            Initialized State
        """
        bias = np.zeros((natoms, 3))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        force_sum = np.zeros((*grid.shape, dims))
        force = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        return ABFState(bias, None, hist, force_sum, force, Wp, Wp_)

    def update(state, data):
        """
        Advance the state of the ABF simulation.

        Arguments
        ---------
        state: ABFstate
            Old ABFstate from the previous simutlation step.
        data: JaxArray
            Snapshot to access simulation data.

        Returns
        -------
        ABFState
            Updated internal state
        """
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)

        p = data.momenta
        # The following could equivalently be computed as `linalg.pinv(Jxi.T) @ p`
        # (both seem to have the same performance).
        # Another option to benchmark against is
        # Wp = linalg.tensorsolve(Jxi @ Jxi.T, Jxi @ p)
        Wp = scipy.linalg.solve(Jxi @ Jxi.T, Jxi @ p, sym_pos="sym")
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt

        I_xi = get_grid_index(xi)
        N_xi = state.hist[I_xi] + 1
        # Add previous force to remove bias
        force_xi = state.force_sum[I_xi] + dWp_dt + state.force
        hist = state.hist.at[I_xi].set(N_xi)
        force_sum = state.force_sum.at[I_xi].set(force_xi)
        force = force_xi / np.maximum(N_xi, N)

        bias = np.reshape(-Jxi.T @ force, state.bias.shape)

        return ABFState(bias, xi, hist, force_sum, force, Wp, state.Wp)

    return snapshot, initialize, generalize(update, helpers)
