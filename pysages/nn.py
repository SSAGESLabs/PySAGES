# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import cupy
import matplotlib.pyplot as plt
import numba
import numpy as onp

import jax
import jax.numpy as np
import jax_md as md

from jax import dlpack
from jax.numpy import linalg
from jax.config import config

config.update("jax_enable_x64", True)

from collections import namedtuple
from functools import partial
from itertools import chain
from operator import itemgetter
from fastcore.dispatch import typedispatch
from jax.scipy.linalg import solve
from jax.lax import while_loop, cond
from pysages.utils import register_pytree_namedtuple
#
import jax.experimental.stax as stax


# %%
def rng_key(seed = 0, n = 2):
    """
    Returns a pseudo-randomly generated key, constructed by calling
    `jax.random.PRNGKey(seed)` and then splitting it `n` times.
    """
    key = jax.random.PRNGKey(seed)
    for _ in range(n):
        key, _ = jax.random.split(key)
    return key
#
def prod(xs):
    y = 1
    for x in xs:
        y *= x
    return y


# %%
#
# Models
# ------
@register_pytree_namedtuple
class MLP(namedtuple("MLP", ("parameters", "apply"))):
    def __repr__(self):
        return repr(type(self).__name__ + " object with fields: (parameters, apply)")
#
def mlp(indim, outdim, topology, activation = stax.Tanh, seed = 0):
    """
    Returns a MLP object, which contains the parameters of a
    multilayer-perceptron network with inner layers defined by `topology`,
    and activation function `σ` (defaults to `stax.Tanh`).
    """
    σ = activation
    # Concatenate inner layers and activation functions
    layers = list(chain.from_iterable((stax.Dense(i), σ) for i in topology))
    # Add flattening and output layers
    layers = [stax.Flatten] + layers + [stax.Dense(outdim)]
    # Build initialization and application functions for the network
    init, apply = stax.serial(*layers)
    # Randomly initialize network parameters with seed
    _, parameters = init(rng_key(seed), np.zeros(indim))
    #
    return MLP(parameters, apply)
#
def unpack(params):
    data, structure = jax.tree_flatten(params)
    shapes = [entry.shape for entry in data]
    s = (prod(shapes[-1]), -1)
    ps = np.hstack([np.reshape(entry, s) for entry in data])
    return ps, shapes, structure
#
def pack(structure, shapes, params):
    inds = np.cumsum(np.array([prod(s) for s in shapes]))
    part = np.split(params, inds[:-1], axis = 1)
    ps = [p.reshape(s) for (p, s) in zip(part, shapes)]
    return structure.unflatten(ps)


# %%
#
# Objectives, Costs, Regularization
# ---------------------------------
def v_error(model):
    _, shapes, structure = unpack(model.parameters)
    def compute(p, inputs, reference):
        "Vectorized (pointwise) error"
        params = pack(structure, shapes, p)
        prediction = model.apply(params, inputs)
        return np.float32(prediction - reference)
    return compute
#
def sum_squares(errors):
    return np.sum(errors**2, dtype = np.float64) / 2
#
def l2_norm(θ):
    return (θ.flatten() @ θ.flatten().T) / 2
#
@register_pytree_namedtuple
class HyperParameters(
    namedtuple("HyperParameters", ("α", "β"), defaults = (np.float32(1), np.float32(1)))
):
    pass
#
@register_pytree_namedtuple
class PartialRObjective(
    namedtuple(
        "PartialRObjective",
        ("cost", "regularizer", "hyperparameters", "max_iters"),
        defaults = (
            sum_squares,
            l2_norm,
            HyperParameters(),
            500
        )
    )
):
    pass
#
@register_pytree_namedtuple
class PartialRBObjective(
    namedtuple(
        "PartialRObjective",
        ("cost", "regularizer", "max_iters"),
        defaults = (
            sum_squares,
            l2_norm,
            500
        )
    )
):
    pass
#
@register_pytree_namedtuple
class RObjective(
    namedtuple(
        "RObjective",
        ("errors", "cost", "regularizer", "hyperparameters", "max_iters")
    )
):
    pass
#
@register_pytree_namedtuple
class RBObjective(
    namedtuple(
        "RObjective",
        ("errors", "cost", "regularizer", "max_iters")
    )
):
    pass
#
@typedispatch
def initialize(o: PartialRBObjective, model):
    errors = v_error(model)
    return RBObjective(errors, o.cost, o.regularizer, o.max_iters)


# %%
#
# Optimizers
# ----------
@register_pytree_namedtuple
class Optimizer(namedtuple("Optimizer", ("init", "condition", "update"))):
    pass
#
@register_pytree_namedtuple
class LevenbergMaquardtBayes(
    namedtuple(
        "LevenbergMaquardtBayes",
        ("μi", "μs", "μmin", "μmax"),
        defaults = (
            np.float32(0.005),  # μi
            np.float32(10),     # μs
            np.float32(5e-16),  # μmin
            np.float32(1e10)    # μmax
        )
    )
):
    pass


# %%
#
# Training
# --------
@register_pytree_namedtuple
class LMBTrainingState(
    namedtuple("LMBTrainingState", ("θ", "e", "G", "C", "R", "Λ", "μ", "τ", "i", "k"))
):
    pass
#
@register_pytree_namedtuple
class LMState(namedtuple("LMState", ["θ", "e", "G", "C", "R", "μ"])):
    pass
#
@typedispatch
def initialize(opt: LevenbergMaquardtBayes, obj: RBObjective, x, n):
    errors = obj.errors
    jacobian = jax.jacfwd(errors)
    I = np.eye(n, dtype = np.float32)
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
        p = θ - solve(H + state.μ * I, Je, sym_pos = "sym").T
        e = errors(p, x, y)
        C = obj.cost(e)
        R = obj.regularizer(θ)
        G = np.float32(β * C + α * R)
        return LMState(p, e, G, C, R, state.μ * μs)
    #
    def _bl_update(H, C, R, state):
        G, (α, _), μ, τ = state
        tr_inv_H = np.trace(solve(H, I, sym_pos = "sym"))
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
    obj = initialize(objective, model)
    opt = initialize(optimizer, obj, inputs, n)
    #
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
