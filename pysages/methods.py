# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020: SSAGES Team (see LICENSE.md)


from collections import namedtuple

import jax.numpy as np
from jax import jit, pmap, vmap
from jax.numpy import linalg
from jax.ops import index, index_add, index_update

from .grids import get_index
from .utils import register_pytree_namedtuple


ABFData = namedtuple(
    "ABFData",
    ["bias", "hist", "Fsum", "F", "Wp", "Wp_"]
)

FUNNData = namedtuple(
    "FUNNData",
    ["bias", "nn", "hist", "Fsum", "F", "Wp", "Wp_"]
)


@register_pytree_namedtuple
class ABFState(ABFData):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


@register_pytree_namedtuple
class FUNNState(FUNNData):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


def check_dims(grid, cv):
    if grid.shape.size != cv.dims:
        raise ValueError("Grid and Collective Variable dimensions must match.")


def generic_update(cv, concrete_update):
    _update = jit(concrete_update)
    #
    def update(snapshot, state):
        M = snapshot.vel_mass[:, 3:]
        V = snapshot.vel_mass
        # Compute the collective variable
        ξ, Jξ = cv.ξ(snapshot.positions, snapshot.tags)
        #
        return _update(M, V, ξ, Jξ, state)
    #
    return update


def abf(snapshot, grid, cv, N = 200):
    check_dims(grid, cv)
    #
    N = np.asarray(N)
    dt = snapshot.dt
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
    def update(M, V, ξ, Jξ, state):
        # Compute momenta
        p = np.multiply(M, V).flatten()
        # The following could equivalently be computed as `linalg.pinv(Jξ.T) @ p`
        # (both seem to have the same performance).
        Wp = linalg.tensorsolve(Jξ @ Jξ.T, Jξ @ p)
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt + state.F
        #
        I = get_index(grid, ξ)
        H_I = state.hist[I] + 1
        ΣF_I = state.Fsum[I] + dWp_dt
        hist = state.hist.at[I].set(H_I)
        Fsum = state.Fsum.at[I].set(ΣF_I)
        F = ΣF_I / np.maximum(H_I, N)
        #
        bias = np.reshape(-Jξ.T @ F, snapshot.forces.shape)
        #
        return ABFState(bias, hist, Fsum, F, Wp, state.Wp)
    #
    return snapshot, initialize, generic_update(cv, update)


def funn(snapshot, grid, cv, topology, N = 200):
    check_dims(grid, cv)
    #
    N = np.asarray(N)
    dt = snapshot.dt
    #
    def initialize():
        bias = np.zeros_like(snapshot.forces)
        hist = np.zeros(grid.shape, dtype = np.uint32)
        Fsum = np.zeros(np.hstack([grid.shape, cv.dims]))
        F = np.zeros(cv.dims)
        Wp = np.zeros(cv.dims)
        Wp_ = np.zeros(cv.dims)
        nn = neural_network(topology)
        return FUNNState(bias, nn, hist, Fsum, F, Wp, Wp_)
    #
    def update(M, V, ξ, Jξ, state):
        #
        nn.train()
        Q = nn.predict(Fsum)
        #
        # Compute momenta
        p = np.multiply(M, V).flatten()
        Wp = linalg.tensorsolve(Jξ @ Jξ.T, Jξ @ p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp_ + 0.5 * state.Wp_) / dt + state.F + Q
        #
        I = get_index(grid, ξ)
        H_I = state.hist[I] + 1
        ΣF_I = state.Fsum[I] + dWp_dt
        hist = state.hist.at[I].set(H_I)
        Fsum = state.Fsum.at[I].set(ΣF_I)
        F = ΣF_I / np.maximum(H_I, N)
        #
        bias = np.reshape(-Jξ.T @ F, snapshot.forces.shape)
        #
        return FUNNState(bias, nn, hist, Fsum, F, Wp, Wp_)
    #
    return snapshot, initialize, generic_update(cv, update)
