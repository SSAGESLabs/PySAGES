# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020: SSAGES Team (see LICENSE.md)

from collections import namedtuple

import jax.numpy as np
from jax import jit, pmap, vmap, scipy
from jax.numpy import linalg
from jax.ops import index, index_add, index_update
from pysages.ssages.cvs import collective_variable
from pysages.nn.models import mlp
from pysages.nn.objectives import PartialRBObjective
from pysages.nn.optimizers import LevenbergMaquardtBayes
from pysages.nn.training import trainer
from pysages.utils import register_pytree_namedtuple

from .grids import get_index


def check_dims(grid, cv):
    if grid.shape.size != cv.dims:
        raise ValueError("Grid and Collective Variable dimensions must match.")


def generic_update(concrete_update):
    _update = jit(concrete_update)
    #
    def update(snapshot, state):
        VM = snapshot.vel_mass
        R = snapshot.positions
        T = snapshot.ids
        #
        return _update(VM, R, T, state)
    #
    # TODO: Validate that jitting here gives correct results
    return jit(update)


class ABF:
    def __init__(self, grid, cv, N = 200):
        ξ = collective_variable(*cv)
        check_dims(grid, ξ)

        self.grid = grid
        self.cv = ξ
        self.N = np.asarray(N)
    #
    def __call__(self, snapshot, helpers):
        return abf(snapshot, self.grid, self.cv, self.N, helpers)


class ABFState(
    namedtuple(
        "ABFState",
        ("bias", "hist", "Fsum", "F", "Wp", "Wp_"),
    )
):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


def abf(snapshot, grid, cv, N, helpers):
    dt = snapshot.dt
    dims = grid.shape.size
    indices, momenta = helpers
    #
    def initialize():
        bias = np.zeros_like(snapshot.forces)
        hist = np.zeros(grid.shape, dtype = np.uint32)
        Fsum = np.zeros(np.hstack([grid.shape, cv.dims]))
        F = np.zeros(cv.dims)
        Wp = np.zeros(cv.dims)
        Wp_ = np.zeros(cv.dims)
        return ABFState(bias, hist, Fsum, F, Wp, Wp_)
    #
    def update(VM, R, T, state):
        # Compute the collective variable and its jacobian
        ξ, Jξ = cv(R, indices(T))
        #
        p = momenta(VM)
        # The following could equivalently be computed as `linalg.pinv(Jξ.T) @ p`
        # (both seem to have the same performance).
        # Another option to benchmark against is
        # Wp = linalg.tensorsolve(Jξ @ Jξ.T, Jξ @ p)
        Wp = scipy.linalg.solve(Jξ @ Jξ.T, Jξ @ p, sym_pos = "sym")
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I = get_index(grid, ξ)
        H_I = state.hist[I] + 1
        # Add previous force to remove bias
        ΣF_I = state.Fsum[I] + dWp_dt + state.F
        hist = state.hist.at[I].set(H_I)
        Fsum = state.Fsum.at[I].set(ΣF_I)
        F = ΣF_I / np.maximum(H_I, N)
        #
        bias = np.reshape(-Jξ.T @ F, snapshot.forces.shape)
        #
        return ABFState(bias, hist, Fsum, F, Wp, state.Wp)
    #
    return snapshot, initialize, generic_update(update)


class FUNN:
    def __init__(self, grid, cv, topology, N = 200):
        ξ = collective_variable(*cv)
        check_dims(grid, ξ)

        self.grid = grid
        self.cv = ξ
        self.topology = topology
        self.N = np.asarray(N)
    #
    def __call__(self, snapshot, helpers):
        return funn(snapshot, self.grid, self.cv, self.topology, self.N, helpers)


class FUNNState(
    namedtuple(
        "FUNNState",
        ("bias", "nn", "hist", "Fsum", "F", "Wp", "Wp_"),
    )
):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


def funn(snapshot, grid, cv, topology, N, helpers):
    dt = snapshot.dt
    dims = grid.shape.size
    indices, momenta = helpers
    model = mlp(grid.shape, dims, topology)
    train = trainer(model, PartialRBObjective(), LevenbergMaquardtBayes(), np.zeros(cv.dims))
    #
    def initialize():
        bias = np.zeros_like(snapshot.forces)
        hist = np.zeros(grid.shape, dtype = np.uint32)
        Fsum = np.zeros(np.hstack([grid.shape, cv.dims]))
        F = np.zeros(cv.dims)
        Wp = np.zeros(cv.dims)
        Wp_ = np.zeros(cv.dims)
        return FUNNState(bias, model.parameters, hist, Fsum, F, Wp, Wp_)
    #
    def update(VM, R, T, state):
        # Compute the collective variable and its jacobian
        ξ, Jξ = cv(R, indices(T))
        #
        θ = train(state.nn, state.Fsum / state.hist).θ
        Q = model.apply(θ, ξ)
        #
        p = momenta(VM)
        Wp = linalg.tensorsolve(Jξ @ Jξ.T, Jξ @ p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp_ + 0.5 * state.Wp_) / dt
        #
        I = get_index(grid, ξ)
        H_I = state.hist[I] + 1
        ΣF_I = state.Fsum[I] + dWp_dt + Q
        hist = state.hist.at[I].set(H_I)
        Fsum = state.Fsum.at[I].set(ΣF_I)
        F = ΣF_I / np.maximum(H_I, N)
        #
        bias = np.reshape(-Jξ.T @ F, snapshot.forces.shape)
        #
        return FUNNState(bias, θ, hist, Fsum, F, Wp, state.Wp)
    #
    return snapshot, initialize, generic_update(update)
