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
    ["snapshot", "bias", "grid", "hist", "Fsum", "F", "Wp", "Wp_", "N", "dt"]
)

FUNNData = namedtuple(
    "FUNNData",
    ["snapshot", "bias", "grid", "nn", "hist", "Fsum", "F", "Wp", "Wp_", "N", "dt"]
)


@register_pytree_namedtuple
class ABFState(ABFData):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


@register_pytree_namedtuple
class FUNNState(FUNNData):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


def abf(snapshot, grid, cv, N = 200):
    N = np.asarray(N)
    #
    def initialize():
        bias = np.zeros_like(snapshot.forces)
        hist = np.zeros(grid.shape, dtype = np.uint32)
        Fsum = np.zeros(grid.shape)
        F = np.zeros(cv.dims)
        Wp = np.zeros(cv.dims)
        Wp_ = np.zeros(cv.dims)
        dt = snapshot.dt
        return ABFState(snapshot, bias, grid, hist, Fsum, F, Wp, Wp_, N, dt)
    #
    def update(state, timestep):
        snapshot, bias, grid, hist, Fsum, F, Wp_, Wp__, N, dt = state
        M = snapshot.vel_mass[:, 3:4]
        V = snapshot.vel_mass
        # Compute the collective variable
        ξ, Jξ = cv.ξ(snapshot.positions, snapshot.tags)
        # Compute momenta
        p = np.multiply(M, V).flatten()
        # The following could equivalently be computed by np.pinv(Jξ.transpose()) @ p,
        # but it does not work when ξ is scalar.
        Wp = linalg.tensorsolve(Jξ @ Jξ.transpose(), Jξ @ p)
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2 * Wp_ + 0.5 * Wp__) / dt + F
        #
        I = get_index(grid, ξ)
        hist = hist.at[I].add(1)
        Fsum = Fsum.at[I].add(dWp_dt)
        n = hist[I]
        F = dWp_dt / np.maximum(n, N)
        #
        bias = np.reshape(-Jξ.transpose() @ F, snapshot.forces.shape)
        #
        return ABFState(snapshot, bias, grid, hist, Fsum, F, Wp, Wp_, N, dt)
    #
    return jit(initialize), jit(update)


def funn(snapshot, grid, topology, cv, N = 200):
    N = np.asarray(N)
    #
    def initialize():
        bias = np.zeros_like(snapshot.forces)
        hist = np.zeros(grid.shape, dtype = np.uint32)
        Fsum = np.zeros(grid.shape)
        F = np.zeros(cv.dims)
        Wp = np.zeros(cv.dims)
        Wp_ = np.zeros(cv.dims)
        nn = neural_network(topology)
        dt = snapshot.dt
        return FUNNState(snapshot, bias, grid, nn, hist, Fsum, F, Wp, Wp_, N, dt)
    #
    def update(state, ξ, Jξ):
        snapshot, grid, hist, Fsum, F, Wp_, Wp__, N, dt = state
        M = snapshot.vel_mass[:, 3:4]
        V = snapshot.vel_mass
        #
        nn.train()
        Q = nn.predict(Fsum)
        #
        p = np.multiply(M, V).flatten()
        Wp = linalg.tensorsolve(jξ @ Jξ.transpose(), Jξ @ p)
        dWp_dt = (1.5 * Wp - 2 * Wp_ + 0.5 * Wp__) / dt + F + Q
        #
        I = get_index(grid, ξ)
        hist = hist.at[I].add(1)
        Fsum = Fsum.at[I].add(dWp_dt)
        n = hist[I]
        F = dWp_dt / np.maximum(n, N)
        #
        bias = -Jξ.transpose() @ (F / np.max(n, N))
        #
        return FUNNState(snapshot, bias, grid, nn, hist, Fsum, F, Wp, Wp_, N, dt)
    #
    return jit(initialize), jit(update)
