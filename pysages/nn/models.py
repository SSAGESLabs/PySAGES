# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import jax.experimental.stax as stax
import jax.numpy as np

from collections import namedtuple
from itertools import chain
from pysages.utils import register_pytree_namedtuple

from .utils import rng_key


@register_pytree_namedtuple
class MLP(namedtuple("MLP", ("parameters", "apply"))):
    def __repr__(self):
        return repr(type(self).__name__ + " object with fields: (parameters, apply)")


def mlp(indim, outdim, topology, activation=stax.Tanh, seed=0):
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
