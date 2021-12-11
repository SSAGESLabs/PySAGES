# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import NamedTuple

from jax import value_and_grad, vmap
from plum import Dispatcher

from pysages.ml.utils import (
    number_of_weights,
    pack,
    prod,
    sum_squares,
    unpack,
)
from pysages.utils import Float

import jax.numpy as np
import numpy as onp


# Create a dispatcher for this submodule
dispatch = Dispatcher()


# Losses
class Loss:
    """
    Abstract base class for all losses.
    """
    pass


class GradientsLoss(Loss):
    """
    Abstract base class for gradient-based losses.
    """
    pass


class Sobolev1Loss(Loss):
    """
    Abstract base class for losses that depend on both target values and gradients.
    """
    pass


class SSE(Loss):
    """
    Sum-of-Squared-Errors Loss.
    """
    pass


class GradientsSSE(GradientsLoss):
    """
    Sum-of-Squared-Gradient-Errors Loss.
    """
    pass


class Sobolev1SSE(Sobolev1Loss):
    """
    Sum-of-Squared-Errors and Squared-Gradient-Errors Loss.
    """
    pass


# Regularizers
class Regularizer:
    """
    Abstract base class for all regularizers.
    """
    pass


# On Python >= 3.9 NamedTuple cannot be used directly as superclass
L2Regularization = NamedTuple("L2Regularization", [("coeff", Float)])


class L2Regularization(Regularizer, L2Regularization):
    """
    L2-norm regularization.

    coeff: Float
        Hyperparameter, coefficient for the regularizing term.
    """


class VarRegularization(Regularizer):
    """
    Weights-variance regularization.
    """
    pass


@dispatch.abstract
def build_objective_function(model, loss, reg):
    """
    Given a model, loss and regularizer, it builds an objective function that takes
    a set of parameters, input and reference values for the model, and returns the
    corresponding value for the objective.
    """


@dispatch
def build_objective_function(model, loss: Loss, reg: Regularizer):
    cost = build_cost_function(loss, reg)

    def objective(params, inputs, reference):
        prediction = model.apply(params, inputs).reshape(reference.shape)
        e = np.asarray(prediction - reference, dtype = np.float32).flatten()
        ps, _ = unpack(params)
        return cost(e, ps)

    return objective


@dispatch
def build_objective_function(model, loss: Sobolev1Loss, reg: Regularizer):
    apply = value_and_grad(
        lambda p, x: model.apply(p, x.reshape(1, -1)).sum(), argnums = 1
    )
    cost = build_cost_function(loss, reg)

    def objective(params, inputs, refs):
        reference, refgrads = refs
        prediction, gradients = vmap(lambda x: apply(params, x))(inputs)
        prediction = prediction.reshape(reference.shape)
        gradients = gradients.reshape(refgrads.shape)
        e = np.asarray(prediction - reference, dtype = np.float32).flatten()
        ge = np.asarray(gradients - refgrads, dtype = np.float32).flatten()
        ps, _ = unpack(params)
        return cost((e, ge), ps)

    return objective


@dispatch.abstract
def build_cost_function(loss, reg):
    """
    Given a loss and regularizer, it builds a regularized cost function that takes
    a set of model parameters and the error differences between such model's
    predictions at some target values.
    """


@dispatch
def build_cost_function(loss: SSE, reg: L2Regularization):
    r = reg.coeff

    def cost(errors, ps):
        return (sum_squares(errors) + r * sum_squares(ps)) / 2

    return cost


@dispatch
def build_cost_function(loss: SSE, reg: VarRegularization):

    def cost(errors, ps):
        # k = ps.size
        return (sum_squares(errors) + ps.var()) / 2

    return cost


@dispatch
def build_cost_function(loss: Sobolev1SSE, reg: L2Regularization):
    r = reg.coeff

    def cost(errors, ps):
        e, ge = errors
        return (sum_squares(e) + sum_squares(ge) + r * sum_squares(ps)) / 2

    return cost


@dispatch
def build_cost_function(loss: Sobolev1SSE, reg: VarRegularization):

    def cost(errors, ps):
        # k = ps.size
        e, ge = errors
        return (sum_squares(e) + sum_squares(ge) + ps.var()) / 2

    return cost


@dispatch.abstract
def build_error_function(model, loss):
    """
    Given a model and loss, it builds a function that computes the error
    differences between the model's predictions at each input value and some
    reference values.
    """


@dispatch
def build_error_function(model, loss: Loss):
    _, layout = unpack(model.parameters)

    def error(ps, inputs, reference):
        params = pack(ps, layout)
        prediction = model.apply(params, inputs).reshape(reference.shape)
        return np.asarray(prediction - reference, dtype = np.float32).flatten()

    return error


@dispatch
def build_error_function(model, loss: Sobolev1Loss):
    apply = value_and_grad(
        lambda p, x: model.apply(p, x.reshape(1, -1)).sum(), argnums = 1
    )
    _, layout = unpack(model.parameters)

    def error(ps, inputs, refs):
        params = pack(ps, layout)
        reference, refgrads = refs
        # prediction, gradients = apply(params, inputs)
        # gradients = grad_apply(params, inputs).reshape(refgrads.shape)
        prediction, gradients = vmap(lambda x: apply(params, x))(inputs)
        prediction = prediction.reshape(reference.shape)
        gradients = gradients.reshape(refgrads.shape)
        e = np.asarray(prediction - reference, dtype = np.float32).flatten()
        ge = np.asarray(gradients - refgrads, dtype = np.float32).flatten()
        return (e, ge)

    return error


def build_split_cost_function(model, loss, reg):
    error = build_error_function(model, loss)
    cost = build_cost_function(loss, reg)
    return error, cost


@dispatch.abstract
def build_damped_hessian(loss, reg):
    """
    Returns a function that evaluates the damped hessian for the
    Levenberg-Marquardt optimizer, given a model's Jacobian `J` with respect
    to its parameters, and a damping value mu.
    """


@dispatch
def build_damped_hessian(loss: Loss, reg: L2Regularization):
    r = reg.coeff

    def dhessian(J, mu):
        H = J.T @ J
        i = np.diag_indices_from(H)
        return H.at[i].add(r + mu)

    return dhessian


@dispatch
def build_damped_hessian(loss: Loss, reg: VarRegularization):

    def dhessian(J, mu):
        H = J.T @ J
        k = H.shape[0]
        i = np.diag_indices_from(H)
        return H.at[i].add((1 - 1 / k)**2 / k + mu)

    return dhessian


@dispatch
def build_damped_hessian(loss: Sobolev1Loss, reg: L2Regularization):
    r = reg.coeff

    def dhessian(jacs, mu):
        J, gJ = jacs
        H = J.T @ J + gJ.T @ gJ
        i = np.diag_indices_from(H)
        return H.at[i].add(r + mu)

    return dhessian


@dispatch
def build_damped_hessian(loss: Sobolev1Loss, reg: VarRegularization):

    def dhessian(jacs, mu):
        J, gJ = jacs
        H = J.T @ J + gJ.T @ gJ
        k = H.shape[0]
        i = np.diag_indices_from(H)
        return H.at[i].add((1 - 1 / k)**2 / k + mu)

    return dhessian


@dispatch.abstract
def build_jac_err_prod(loss, reg):
    """
    Returns a function that evaluates the product of a model's Jacobian `J` with
    respect to its parameters, and error differences `e`. Used within the
    Levenberg-Marquardt optimizer.
    """


@dispatch
def build_jac_err_prod(loss: Loss, reg: L2Regularization):
    r = reg.coeff

    def jep(J, e, ps):
        return J.T @ e + r * ps

    return jep


@dispatch
def build_jac_err_prod(loss: Loss, reg: VarRegularization):

    def jep(J, e, ps):
        k = ps.size
        return J.T @ e + (1 - 1 / k) / k * (ps - ps.mean())

    return jep


@dispatch
def build_jac_err_prod(loss: Sobolev1Loss, reg: L2Regularization):
    r = reg.coeff

    def jep(jacs, errors, ps):
        J, gJ = jacs
        e, ge = errors
        return J.T @ e + gJ.T @ ge + r * ps

    return jep


@dispatch
def build_jac_err_prod(loss: Sobolev1Loss, reg: VarRegularization):

    def jep(jacs, errors, ps):
        k = ps.size
        J, gJ = jacs
        e, ge = errors
        return J.T @ e + gJ.T @ ge + (1 - 1 / k) / k * (ps - ps.mean())

    return jep


def estimate_l2_coefficient(topology, grid):
    # Grid dimensionality
    d = grid.shape.size
    # Polynomial degree estimate
    k = onp.ceil(onp.sqrt(grid.shape.sum())) + 1
    # Number of weights for reasonably-sized single-hidden-layer NN
    n = number_of_weights((d, k, d))
    # Number of weights for the chosen topology
    p = number_of_weights((d, *topology, d))
    # If we have too few parameters the regularization term should be
    # negligible, for too many parameters the hyperparameter value
    # `len(topology)**2 / prod(grid.shape)` seems to work fine irrespectively
    # of the number of weights. Hence, we use a sigmoid to estimate the
    # regularization coeffiecient.
    return len(topology)**2 / prod(grid.shape) / (1 + onp.exp((n - p) / 2))
