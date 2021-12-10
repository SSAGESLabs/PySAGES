# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from dataclasses import dataclass
from itertools import chain
from typing import Callable, Optional

from jax import numpy as np
from jax.nn.initializers import variance_scaling
from plum import dispatch

from pysages.ml.utils import rng_key
from pysages.utils import JaxArray, try_import

stax = try_import("jax.example_libraries.stax", "jax.experimental.stax")


@dataclass
class Model:
    """
    Base class for all models. Contains the model parameters and function that,
    given an input, return the approximation defined by the model.
    """
    parameters: JaxArray
    apply:      Callable


@dataclass
class AbstractMLP(Model):

    def __init__(self, indim, layers, seed):
        # Build initialization and application functions for the network
        init, apply = stax.serial(*layers)
        # Randomly initialize network parameters with seed
        _, parameters = init(rng_key(seed), (-1, indim))
        #
        self.parameters = parameters
        self.apply = apply


@dataclass
class MLP(AbstractMLP):
    """
    Multilayer-perceptron network.
    """

    def __init__(
        self, indim, outdim, topology, activation = stax.Tanh, transform = None, seed = 0
    ):
        """
        Arguments
        ---------

        indim:
            Dimensionality of the a single data input.

        outdim:
            Dimensionality of the a single data output.

        topology: Tuple
            Defines the structure of the inner layers for the network.

        activation:
            Activation function from `stax` (defaults to `stax.Tanh`).

        transform: Optional[Callable]
            Optional pre-network transformation function.

        seed: Optional[int]
            Initial seed for weight initialization.
        """

        tlayer = build_transform_layer(transform)
        σ = activation
        # Concatenate inner layers and activation functions
        layers = list(chain.from_iterable((stax.Dense(i), σ) for i in topology))
        # Add output layer
        layers = [stax.Flatten, *tlayer] + layers + [stax.Dense(outdim)]
        #
        super().__init__(indim, layers, seed)


@dataclass
class Siren(AbstractMLP):
    """
    Siren network as decribed in [1]

    [1] Sitzmann, V., Martel, J., Bergman, A., Lindell, D., and Wetzstein, G. "Implicit
    neural representations with periodic activation functions." Advances in Neural
    Information Processing Systems 33 (2020).
    """

    def __init__(
        self, indim, outdim, topology, omega = 1.0, transform = None, seed = 0
    ):
        """
        Arguments
        ---------

        indim:
            Dimensionality of the a single data input.

        outdim:
            Dimensionality of the a single data output.

        topology: Tuple
            Defines the structure of the inner layers for the network.

        omega:
            Weight distribution factor ω₀ for the first layer (as described in [1]).

        transform: Optional[Callable]
            Optional pre-network transformation function.

        seed: Optional[int]
            Initial seed for weight initialization.
        """

        tlayer = build_transform_layer(transform)
        # Weight initialization for Sirens
        pdf_in = variance_scaling(1.0 / 3, "fan_in", "uniform")
        pdf = variance_scaling(2.0 / omega**2, "fan_in", "uniform")
        # Sine activation function
        σ_in = stax.elementwise(lambda x: np.sin(omega * x))
        σ = stax.elementwise(lambda x: np.sin(x))
        # Build layers
        layer_in = [stax.Flatten, *tlayer, stax.Dense(topology[0], pdf_in), σ_in]
        layers = list(chain.from_iterable(
            (stax.Dense(i, pdf), σ) for i in topology[1:]
        ))
        layers = layer_in + layers + [stax.Dense(outdim, pdf)]
        #
        super().__init__(indim, layers, seed)


@dispatch
def build_transform_layer(transform: Optional[Callable] = None):
    return () if transform is None else (stax.elementwise(transform),)
