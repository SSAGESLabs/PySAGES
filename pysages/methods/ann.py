# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from collections import namedtuple
from pysages.nn.models import mlp
from pysages.nn.objectives import PartialRBObjective
from pysages.nn.optimizers import LevenbergMaquardtBayes
from pysages.nn.training import trainer
from pysages.grids import build_indexer

from .core import NNSamplingMethod, generalize  # pylint: disable=relative-beyond-top-level

import jax.numpy as np


# ======= #
#   ANN   #
# ======= #

class ANNState(namedtuple(
    "ANNState",
    (
        "bias",
        "nn",
        "hist",
        "uhist",
        "weight",
        "weight_"
    )
)):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class ANN(NNSamplingMethod):
    snapshot_flags = {"positions", "indices"}

    def build(self, snapshot, helpers):
        self.N = np.asarray(self.kwargs.get('N', 200))
        return _ann(self, snapshot, helpers)


def _ann(method, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    topology = method.topology

    kBT = snapshot.kBT
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)
    get_grid_index = build_indexer(grid)
    model = mlp(grid.shape, dims, topology)
    train = trainer(model, PartialRBObjective(),
                    LevenbergMaquardtBayes(), np.zeros(dims))

    def initialize():
        bias = np.zeros((natoms, dims))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        uhist = hist.copy()
        weight = np.ones(1)
        weight_ = np.zeros(1)
        return ANNState(bias, model.parameters, hist, uhist, weight, weight_)

    def update(state, data):
        # Compute the collective variable and its jacobian
        ξ, Jξ = cv(data)
        #
        θ = train(state.nn, state.bias).θ
        F = model.apply(θ, ξ)
        #
        I_ξ = get_grid_index(ξ)
        N_ξ = state.hist[I_ξ] + 1
        H_ξ = state.weight_ * state.uhist[I_ξ] + \
            state.weight * np.exp(state.bias[I_ξ] / kBT)
        hist = state.hist.at[I_ξ].set(N_ξ)
        uhist = state.uhist.at[I_ξ].set(H_ξ)
        bias = kBT * np.log(H_ξ)
        bias = bias - bias.min()
        #
        bias = np.reshape(-Jξ.T @ F, state.bias.shape)
        #
        return ANNState(bias, θ, hist, uhist, state.weight, state.weight_)

    return snapshot, initialize, generalize(update, helpers)
