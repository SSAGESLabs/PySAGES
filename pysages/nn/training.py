# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES


import jax
import jax.numpy as np

from collections import namedtuple
from functools import partial
from fastcore.dispatch import typedispatch
from jax.lax import while_loop, cond
from jax.scipy.linalg import solve
from pysages.utils import register_pytree_namedtuple

from . import objectives
from . import optimizers
from .objectives import RBObjective
from .optimizers import LevenbergMaquardtBayes, Optimizer
from .utils import unpack


@register_pytree_namedtuple
class LMBTrainingState(
    namedtuple(
        "LMBTrainingState",
        ("θ", "e", "G", "C", "R", "Λ", "μ", "τ", "i", "k")
    )
):
    pass


@register_pytree_namedtuple
class LMState(namedtuple("LMState", ("θ", "e", "G", "C", "R", "μ"))):
    pass


@typedispatch
def initialize(opt: LevenbergMaquardtBayes, obj: RBObjective, x, n):
    errors = obj.errors
    jacobian = jax.jacfwd(errors)
    I = np.eye(n, dtype=np.float32)
    # Bayesian regularization progress parameter
    τ = np.int32(1)
    # Bayesian regularization hyperparameters
    α, β = np.float32(np.minimum(τ / x.size, 1) / x.size), np.float32(1)
    # Levenberg-Maquardt parameters
    μi, μs, μmin, μmax = opt

    #
    def differentiate(θ, e, y):
        J = np.squeeze(jacobian(θ, x, y))
        H = β * J.T @ J + α * I
        Je = β * J.T @ e + α * θ.T
        return H, Je

    #
    def _lm_cond(G, state):
        return (G <= state.G) & (state.μ <= μmax)

    #
    def _lm_update(θ, H, Je, y, Λ, state):
        α, β = Λ
        p = θ - solve(H + state.μ * I, Je, sym_pos="sym").T
        e = errors(p, x, y)
        C = obj.cost(e)
        R = obj.regularizer(θ)
        G = np.float32(β * C + α * R)
        return LMState(p, e, G, C, R, state.μ * μs)

    #
    def _bl_update(H, C, R, state):
        G, (α, _), μ, τ = state
        tr_inv_H = np.trace(solve(H, I, sym_pos="sym"))
        γ = n - α * tr_inv_H
        α = np.float32(n / (2 * R + tr_inv_H))
        β = np.float32((x.shape[0] - γ) / (2 * C))
        return G, (α, β), μ, τ

    #
    def _bl_restart(G, state):
        _, _, _, τ = state
        α = np.float32(np.minimum(τ / x.size, 1) / x.size)
        β = np.float32(1)
        return G, (α, β), μi, τ + 1

    #
    def init(θ, data):
        y = data[1]
        e = errors(θ, x, y)
        i = np.int32(0)  # Iterations counter
        k = np.int32(1)  # Convergence counter
        C = obj.cost(e)
        R = obj.regularizer(θ)
        G = np.float32(β * C + α * R)  # Objective function
        return LMBTrainingState(θ, e, G, C, R, (α, β), μi, τ, i, k)

    #
    def condition(state):
        return (state.k < 4) & (state.i < obj.max_iters)

    #
    def update(data, state):
        y = data[1]
        H, Je = differentiate(state.θ, state.e, y)
        # Inner Levenberg-Maquardt update
        lm_state = LMState(state.θ, state.e, state.G, state.C, state.R, state.μ)
        lm_cond = partial(_lm_cond, state.G)
        lm_update = partial(_lm_update, state.θ, H, Je, y, state.Λ)
        θ, e, G, C, R, μ = while_loop(
            lm_cond,
            lm_update,
            lm_state
        )
        μ = np.where(μ < μmax, μ / μs, μ)
        μ = np.where(μmin < μ, μ, μmin)
        # Bayesian hyperparameter learning
        bl_state = (G, state.Λ, μ, state.τ)
        bl_update = partial(_bl_update, H, C, R)
        bl_restart = partial(_bl_restart, state.G)
        G, Λ, μ, τ = cond(
            G > state.G,
            bl_state,
            bl_restart,
            bl_state,
            bl_update
        )
        k = np.where(G >= state.G, state.k + 1, np.int32(1))
        return LMBTrainingState(θ, e, G, C, R, Λ, μ, τ, state.i + 1, k)

    #
    return Optimizer(init, condition, update)


# %%
def trainer(model, objective, optimizer, inputs):
    # Parameters
    p = unpack(model.parameters)[0]
    n = p.size  # Number of network parameters
    #
    obj = objectives.initialize(objective, model)
    opt = initialize(optimizer, obj, inputs, n)

    # Trainer state initialization
    # ----------------------------
    def train(params, data):
        state = opt.init(params, data)
        update = partial(opt.update, data)
        # Instead of `while`, use `jax.jit`-compatible `while_loop`
        state = while_loop(
            opt.condition,
            update,
            state
        )
        return state

    #
    return jax.jit(train)
