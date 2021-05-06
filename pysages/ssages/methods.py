# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020: SSAGES Team (see LICENSE.md)

import jax.numpy as np

from abc import ABC, abstractmethod
from collections import namedtuple
from jax import jit, scipy
from jax.numpy import linalg
from plum import dispatch
from pysages.ssages.cvs import build
from pysages.nn.models import mlp
from pysages.nn.objectives import PartialRBObjective
from pysages.nn.optimizers import LevenbergMaquardtBayes
from pysages.nn.training import trainer

from .grids import get_index


# ================ #
#   Base Classes   #
# ================ #

class SamplingMethod(ABC):
    def __init__(self, cvs, *args, **kwargs):
        self.cv = build(*cvs)
        self.args = args
        self.kwargs = kwargs
    #
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class GriddedSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, *args, **kwargs):
        check_dims(cvs, grid)
        self.cv = build(*cvs)
        self.grid = grid
        self.args = args
        self.kwargs = kwargs
    #
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class NNSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, topology, *args, **kwargs):
        check_dims(cvs, grid)
        self.cv = build(*cvs)
        self.grid = grid
        self.topology = topology
        self.args = args
        self.kwargs = kwargs
    #
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


# ========= #
#   Utils   #
# ========= #

def check_dims(cvs, grid):
    if len(cvs) != grid.shape.size:
        raise ValueError("Grid and Collective Variable dimensions must match.")


def generalize(concrete_update):
    _update = jit(concrete_update)
    #
    def update(snapshot, state):
        vms = snapshot.vel_mass
        rs = snapshot.positions
        ids = snapshot.ids
        #
        return _update(state, rs, vms, ids)
    #
    # TODO: Validate that jitting here gives correct results
    return jit(update)


# ======= #
#   ABF   #
# ======= #

class ABFState(
    namedtuple(
        "ABFState",
        ("bias", "hist", "Fsum", "F", "Wp", "Wp_"),
    )
):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class ABF(GriddedSamplingMethod):
    def __call__(self, snapshot, helpers):
        N = np.asarray(self.kwargs.get('N', 200))
        return abf(snapshot, self.cv, self.grid, N, helpers)


def abf(snapshot, cv, grid, N, helpers):
    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.forces, 0)
    indices, momenta = helpers
    #
    def initialize():
        bias = np.zeros((natoms, 3))
        hist = np.zeros(grid.shape, dtype = np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        return ABFState(bias, hist, Fsum, F, Wp, Wp_)
    #
    def update(state, rs, vms, ids):
        # Compute the collective variable and its jacobian
        ξ, Jξ = cv(rs, indices(ids))
        #
        p = momenta(vms)
        # The following could equivalently be computed as `linalg.pinv(Jξ.T) @ p`
        # (both seem to have the same performance).
        # Another option to benchmark against is
        # Wp = linalg.tensorsolve(Jξ @ Jξ.T, Jξ @ p)
        Wp = scipy.linalg.solve(Jξ @ Jξ.T, Jξ @ p, sym_pos = "sym")
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_ξ = get_index(grid, ξ)
        N_ξ = state.hist[I_ξ] + 1
        # Add previous force to remove bias
        F_ξ = state.Fsum[I_ξ] + dWp_dt + state.F
        hist = state.hist.at[I_ξ].set(N_ξ)
        Fsum = state.Fsum.at[I_ξ].set(F_ξ)
        F = F_ξ / np.maximum(N_ξ, N)
        #
        bias = np.reshape(-Jξ.T @ F, state.bias.shape)
        #
        return ABFState(bias, hist, Fsum, F, Wp, state.Wp)
    #
    return snapshot, initialize, generalize(update)


# ======== #
#   FUNN   #
# ======== #

class FUNN(NNSamplingMethod):
    def __call__(self, snapshot, helpers):
        N = np.asarray(self.kwargs.get('N', 200))
        return funn(snapshot, self.cv, self.grid, self.topology, N, helpers)


class FUNNState(
    namedtuple(
        "FUNNState",
        ("bias", "nn", "hist", "Fsum", "F", "Wp", "Wp_"),
    )
):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


def funn(snapshot, cv, grid, topology, N, helpers):
    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.forces, 0)
    indices, momenta = helpers
    model = mlp(grid.shape, dims, topology)
    train = trainer(model, PartialRBObjective(), LevenbergMaquardtBayes(), np.zeros(dims))
    #
    def initialize():
        bias = np.zeros((natoms, dims))
        hist = np.zeros(grid.shape, dtype = np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        return FUNNState(bias, model.parameters, hist, Fsum, F, Wp, Wp_)
    #
    def update(state, rs, vms, ids):
        # Compute the collective variable and its jacobian
        ξ, Jξ = cv(rs, indices(ids))
        #
        θ = train(state.nn, state.Fsum / state.hist).θ
        Q = model.apply(θ, ξ)
        #
        p = momenta(vms)
        Wp = linalg.tensorsolve(Jξ @ Jξ.T, Jξ @ p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_ξ = get_index(grid, ξ)
        N_ξ = state.hist[I_ξ] + 1
        F_ξ = state.Fsum[I_ξ] + dWp_dt + Q
        hist = state.hist.at[I_ξ].set(N_ξ)
        Fsum = state.Fsum.at[I_ξ].set(F_ξ)
        F = F_ξ / np.maximum(N_ξ, N)
        #
        bias = np.reshape(-Jξ.T @ F, state.bias.shape)
        #
        return FUNNState(bias, θ, hist, Fsum, F, Wp, state.Wp)
    #
    return snapshot, initialize, generalize(update)
