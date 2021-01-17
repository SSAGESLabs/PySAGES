# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import jax.numpy as np

from collections import namedtuple
from fastcore.dispatch import typedispatch
from pysages.utils import register_pytree_namedtuple

from .utils import l2_norm, sum_squares, v_error


@register_pytree_namedtuple
class HyperParameters(
    namedtuple("HyperParameters", ("α", "β"), defaults=(np.float32(1), np.float32(1)))
):
    pass


@register_pytree_namedtuple
class PartialRObjective(
    namedtuple(
        "PartialRObjective",
        ("cost", "regularizer", "hyperparameters", "max_iters"),
        defaults=(
            sum_squares,
            l2_norm,
            HyperParameters(),
            500
        )
    )
):
    pass


@register_pytree_namedtuple
class PartialRBObjective(
    namedtuple(
        "PartialRObjective",
        ("cost", "regularizer", "max_iters"),
        defaults=(
            sum_squares,
            l2_norm,
            500
        )
    )
):
    pass


@register_pytree_namedtuple
class RObjective(
    namedtuple(
        "RObjective",
        ("errors", "cost", "regularizer", "hyperparameters", "max_iters")
    )
):
    pass


@register_pytree_namedtuple
class RBObjective(
    namedtuple(
        "RObjective",
        ("errors", "cost", "regularizer", "max_iters")
    )
):
    pass


@typedispatch
def initialize(o: PartialRBObjective, model):
    errors = v_error(model)
    return RBObjective(errors, o.cost, o.regularizer, o.max_iters)
