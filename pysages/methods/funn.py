# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from collections import namedtuple
from jax.numpy import linalg
from pysages.nn.models import mlp
from pysages.nn.objectives import PartialRBObjective
from pysages.nn.optimizers import LevenbergMaquardtBayes
from pysages.nn.training import trainer
from pysages.grids import build_indexer

from .core import NNSamplingMethod, generalize  # pylint: disable=relative-beyond-top-level

import jax.numpy as np


# ======== #
#   FUNN   #
# ======== #


class FUNNState(
    namedtuple(
        "FUNNState",
        (
            "bias",
            "nn",
            "hist",
            "Fsum",
            "F",
            "Wp",
            "Wp_",
        ),
    )
):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class FUNN(NNSamplingMethod):
    snapshot_flags = {"positions", "indices", "momenta"}

    def build(self, snapshot, helpers):
        self.N = np.asarray(self.kwargs.get("N", 200))
        return _funn(self, snapshot, helpers)


def _funn(method, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    topology = method.topology

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    get_grid_index = build_indexer(grid)
    model = mlp(grid.shape, dims, topology)
    train = trainer(model, PartialRBObjective(), LevenbergMaquardtBayes(), np.zeros(dims))

    def initialize():
        bias = np.zeros((natoms, dims))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        return FUNNState(bias, model.parameters, hist, Fsum, F, Wp, Wp_)

    def update(state, data):
        # Compute the collective variable and its jacobian
        ξ, Jξ = cv(data)
        #
        θ = train(state.nn, state.Fsum / state.hist).θ
        #
        p = data.momenta
        Wp = linalg.tensorsolve(Jξ @ Jξ.T, Jξ @ p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_ξ = get_grid_index(ξ)
        N_ξ = state.hist[I_ξ] + 1
        F_ξ = state.Fsum[I_ξ] + dWp_dt + state.F
        hist = state.hist.at[I_ξ].set(N_ξ)
        Fsum = state.Fsum.at[I_ξ].set(F_ξ)
        F = model.apply(θ, ξ)  # F_ξ / np.maximum(N_ξ, N)
        #
        bias = np.reshape(-Jξ.T @ F, state.bias.shape)
        #
        return FUNNState(bias, θ, hist, Fsum, F, Wp, state.Wp)

    return snapshot, initialize, generalize(update, helpers)
