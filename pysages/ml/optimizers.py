# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES
from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as np
from jax.lax import cond
from jax.numpy.linalg import pinv

from pysages.ml.objectives import (
    SSE,
    L2Regularization,
    Loss,
    Regularizer,
    build_damped_hessian,
    build_error_function,
    build_jac_err_prod,
    build_objective_function,
    build_split_cost_function,
    sum_squares,
)
from pysages.ml.utils import dispatch, pack, unpack
from pysages.typing import Any, Callable, JaxArray, NamedTuple, Tuple, Union
from pysages.utils import solve_pos_def, try_import

jopt = try_import("jax.example_libraries.optimizers", "jax.experimental.optimizers")


# Optimizers parameters


class AdamParams(NamedTuple):
    """
    Parameters for the ADAM optimizer.
    """

    step_size: Union[float, Callable] = 1e-2
    beta_1: float = 0.9
    beta_2: float = 0.999
    tol: float = 1e-8


class LevenbergMarquardtParams(NamedTuple):
    """
    Parameters for the Levenberg-Marquardt optimizer.
    """

    mu_0: float = 1e-1
    mu_c: float = 10.0
    mu_min: float = 1e-8
    mu_max: float = 1e8
    rho_c: float = 1e-1
    rho_min: float = 1e-4


# Optimizers state


class WrappedState(NamedTuple):
    """
    Holds the data for an optimization run for an optimizer from
    stax.example_libraries.optimizers.
    """

    data: Tuple[JaxArray, JaxArray]
    params: Any
    iters: int = 0
    improved: bool = True


class LevenbergMarquardtState(NamedTuple):
    """
    Holds the data for a Levenberg-Marquardt optimization run.
    """

    data: Tuple[JaxArray, JaxArray]
    params: JaxArray
    errors: JaxArray
    cost: float
    mu: float
    iters: int = 0
    improved: bool = True


class LevenbergMarquardtBRState(NamedTuple):
    """
    Holds the data for a Bayesian-regularized Levenberg-Marquardt optimization run.
    """

    data: Tuple[JaxArray, JaxArray]
    params: JaxArray
    errors: JaxArray
    cost: float
    mu: float
    alpha: float = 1e-4
    iters: int = 0
    improved: bool = True


class Optimizer:
    """
    Abstract base class for all optimizers.
    """

    pass


@dataclass
class Adam(Optimizer):
    """
    ADAM optimizer from stax.example_libraries.optimizers.
    """

    params: AdamParams = AdamParams()
    loss: Loss = SSE()
    reg: Regularizer = L2Regularization(0.0)
    tol: float = 1e-4
    max_iters: int = 10000


@dataclass
class LevenbergMarquardt(Optimizer):
    """
    Levenberg-Marquardt optimizer.
    """

    params: LevenbergMarquardtParams = LevenbergMarquardtParams()
    loss: Loss = SSE()
    reg: Regularizer = L2Regularization(0.0)
    max_iters: int = 500


@dataclass
class LevenbergMarquardtBR(Optimizer):
    """
    Levenberg-Marquardt optimizer with Bayesian regularization.
    """

    params: LevenbergMarquardtParams = LevenbergMarquardtParams()
    alpha: float = 0.0
    max_iters: int = 500
    update: Callable = lambda a, b, c, t: t


@dispatch.abstract
def build(optimizer, model):
    """
    Given an optimizer and a model, builds and return three functions `initialize`,
    `keep_iterating` and `update` that respectively handle the initialization of
    the optimization procedure, evaluate a condition for halting the optimization,
    and update the optimization state.

    See `pysages.ml.training.build_fitting_function` to learn how the returned
    functions are used.
    """


@dispatch
def build(optimizer: Adam, model):
    _init, _update, repack = jopt.adam(*optimizer.params)
    objective = build_objective_function(model, optimizer.loss, optimizer.reg)
    gradient = jax.grad(objective)
    max_iters = optimizer.max_iters
    _, layout = unpack(model.parameters)

    def initialize(params, x, y):
        wrapped_params = _init(pack(params, layout))
        return WrappedState((x, y), wrapped_params)

    def keep_iterating(state):
        return state.improved & (state.iters < max_iters)

    def update(state):
        data, params, iters, _ = state
        dp = gradient(repack(params), *data)
        params = _update(iters, dp, params)
        improved = sum_squares(unpack(dp)[0]) > optimizer.tol
        return WrappedState(data, params, iters + 1, improved)

    return initialize, keep_iterating, update


@dispatch
def build(optimizer: LevenbergMarquardt, model):
    error, cost = build_split_cost_function(model, optimizer.loss, optimizer.reg)
    jac_err_prod = build_jac_err_prod(optimizer.loss, optimizer.reg)
    damped_hessian = build_damped_hessian(optimizer.loss, optimizer.reg)
    jacobian = jax.jacobian(error)
    _, c, mu_min, mu_max, rho_c, rho_min = optimizer.params
    max_iters = optimizer.max_iters

    def initialize(params, x, y):
        e = error(params, x, y)
        mu = optimizer.params.mu_0
        return LevenbergMarquardtState((x, y), params, e, np.inf, mu)

    def keep_iterating(state):
        return state.improved & (state.iters < max_iters) & (state.mu < mu_max)

    def update(state):
        data, p_, e_, C_, mu, iters, _ = state
        x, y = data
        mu = np.float32(mu)
        #
        J = jacobian(p_, x, y)
        H = damped_hessian(J, mu)
        Je = jac_err_prod(J, e_, p_)
        #
        dp = solve_pos_def(H, Je)
        p = p_ - dp
        e = error(p, x, y)
        C = cost(e, p)
        rho = (C_ - C) / (dp.T @ (mu * dp + Je))
        #
        mu = np.where(rho > rho_c, np.maximum(mu / c, mu_min), mu)
        #
        bad_step = (rho < rho_min) | np.any(np.isnan(p))
        mu = np.where(bad_step, np.minimum(c * mu, mu_max), mu)
        p = cond(bad_step, lambda t: t[0], lambda t: t[1], (p_, p))
        e = cond(bad_step, lambda t: t[0], lambda t: t[1], (e_, e))
        C = np.where(bad_step, C_, C)
        improved = (C_ > C) | bad_step
        #
        return LevenbergMarquardtState(data, p, e, C, mu, iters + ~bad_step, improved)

    return initialize, keep_iterating, update


@dispatch
def build(optimizer: LevenbergMarquardtBR, model):
    error = build_error_function(model, SSE())
    jacobian = jax.jacobian(error)
    _, c, mu_min, mu_max, rho_c, rho_min = optimizer.params
    max_iters = optimizer.max_iters
    #
    m = len(model.parameters) / 2 - 1
    k = unpack(model.parameters)[0].size
    update_hyperparams = partial(optimizer.update, m, k, optimizer.alpha)

    def initialize(params, x, y):
        e = error(params, x, y)
        mu = optimizer.params.mu_0
        gamma = np.float64(params.size)
        beta = (x.size / m) ** 2 * (x.size - gamma) / sum_squares(e)
        beta = np.where(beta < 0, 1.0, beta)
        alpha = gamma / sum_squares(params)
        return LevenbergMarquardtBRState((x, y), params, e, np.inf, mu, alpha / beta)

    def keep_iterating(state):
        return state.improved & (state.iters < max_iters) & (state.mu < mu_max)

    def update(state):
        data, p_, e_, C_, mu, alpha, iters, _ = state
        x, y = data
        mu = np.float32(mu)
        alpha_ = np.float32(alpha)
        #
        J = jacobian(p_, x, y)
        H = J.T @ J
        Je = J.T @ e_ + alpha_ * p_
        idx = np.diag_indices_from(H)
        #
        dp = solve_pos_def(H.at[idx].add(alpha_ + mu), Je)
        p = p_ - dp
        e = error(p, x, y)
        C = (sum_squares(e) + alpha * sum_squares(p)) / 2
        rho = (C_ - C) / (dp.T @ (mu * dp + Je))
        #
        mu = np.where(rho > rho_c, np.maximum(mu / c, mu_min), mu)
        #
        bad_step = (rho < rho_min) | np.any(np.isnan(p))
        mu = np.where(bad_step, np.minimum(c * mu, mu_max), mu)
        p = cond(bad_step, lambda t: t[0], lambda t: t[1], (p_, p))
        e = cond(bad_step, lambda t: t[0], lambda t: t[1], (e_, e))
        #
        sse = sum_squares(e)
        ssp = sum_squares(p)
        C = np.where(bad_step, C_, C)
        improved = (C_ > C) | bad_step
        #
        bundle = (alpha, H, idx, sse, ssp, x.size)
        alpha, *_ = cond(bad_step, lambda t: t, update_hyperparams, bundle)
        C = (sse + alpha * ssp) / 2
        #
        return LevenbergMarquardtBRState(data, p, e, C, mu, alpha, iters + ~bad_step, improved)

    return initialize, keep_iterating, update


def update_hyperparams(nlayers, nparams, alpha_0, bundle):
    l, k = nlayers, nparams
    alpha, H, idx, sse, ssp, n = bundle
    gamma = k - alpha * pinv(H.at[idx].add(alpha)).trace()
    reset = np.isnan(gamma) | (gamma >= n) | (sse.sum() < 1e-4) | (ssp.sum() < 1e-4)
    beta = np.where(reset, 1.0, (n / l) ** 2 * (n - gamma) / sse)
    alpha = np.where(reset, alpha_0, gamma / ssp)
    return (alpha / beta, H, idx, sse, ssp, n)
