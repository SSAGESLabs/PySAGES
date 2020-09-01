# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020: SSAGES Team


import jax.numpy as np

from collections import namedtuple
from jax import jit, pmap, vmap
from jax.numpy import linag
from .utils import register_pytree_namedtuple


ABFData = namedtuple(
    "ABFData",
    ["grid", "histogram", "accumulator", "F", "Wp", "Wp_", "N", "dt"]
)

FUNNData = namedtuple(
    "FUNNData",
    ["grid", "nn", "histogram", "accumulator", "F", "Wp", "Wp_", "N", "dt"]
)


@register_pytree_namedtuple
class ABFState(ABFData):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


@register_pytree_namedtuple
class FUNNState(FUNNData):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


def abf(sysview, grid, cv, N = 200):
    def initialize():
        histogram = np.zeros(grid.shape, dtype = uint32)
        accumulator = np.zeros(grid.shape)
        F = np.zeros(cv.dims)
        Wp = np.zeros(cv.dims)
        Wp_ = np.zeros(cv.dims)
        return ABFState(grid, histogram, accumulator, F, Wp, Wp_, N, sysview.dt)
    #
    def update(state, ξ, Jξ, M, p):
        grid, histogram, accumulator, F, Wp_, Wp__, N, dt = state
        #
        p = np.multiply(M, V).flatten()
        Wp = linalg.tensorsolve(jξ @ Jξ.transpose(), Jξ @ p)
        F = (1.5 * Wp - 2 * Wp_ + 0.5 * Wp__) / dt + F
        #
        ind = index(grid, ξ)
        n = histogram[ind]
        histogram[ind] = n + 1
        accumulator[ind] += F
        #
        bias = -Jξ.transpose() @ (F / np.max(n, N))
        #
        return ABFState(grid, histogram, accumulator, F, Wp, Wp_, N, dt)
    #
    return jit(initialize), jit(update)


def funn(sysview, grid, topology, cv, N = 200):
    def initialize():
        histogram = np.zeros(grid.shape, dtype = uint32)
        accumulator = np.zeros(grid.shape)
        F = np.zeros(cv.dims)
        Wp = np.zeros(cv.dims)
        Wp_ = np.zeros(cv.dims)
        nn = neural_network(topology)
        return FUNNState(grid, nn, histogram, accumulator)
    #
    def update(state, ξ, Jξ):
        grid, histogram, accumulator, F, Wp_, Wp__, N, dt = state
        #
        nn.train()
        Q = nn.predict(accumulator)
        #
        p = np.multiply(M, V).flatten()
        Wp = linalg.tensorsolve(jξ @ Jξ.transpose(), Jξ @ p)
        F = (1.5 * Wp - 2 * Wp_ + 0.5 * Wp__) / dt + F + Q
        #
        ind = index(grid, ξ)
        n = histogram[ind]
        histogram[ind] = n + 1
        accumulator[ind] += F
        #
        bias = -Jξ.transpose() @ (F / np.max(n, N))
        #
        return FUNNState(grid, nn, histogram, accumulator, F, Wp, Wp_, N, dt)
    #
    return jit(initialize), jit(update)
