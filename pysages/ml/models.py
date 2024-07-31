# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from dataclasses import dataclass
from itertools import chain

from jax import numpy as np
from jax import random

from pysages.ml.utils import dispatch, rng_key, uniform_scaling
from pysages.typing import Callable, JaxArray, Optional
from pysages.utils import identity, try_import

stax = try_import("jax.example_libraries.stax", "jax.experimental.stax")


@dataclass
class Model:
    """
    Base class for all models. Contains the model parameters and function that,
    given an input, return the approximation defined by the model.
    """

    parameters: JaxArray
    apply: Callable


@dataclass
class MLPBase(Model):
    """
    Multilayer-perceptron network base class. Provides a convenience constructor.
    """

    def __init__(self, indim, layers, seed):
        # Build initialization and application functions for the network
        init, apply = stax.serial(*layers)
        # Randomly initialize network parameters with seed
        _, parameters = init(rng_key(seed), (-1, indim))
        super().__init__(parameters, apply)


@dataclass
class MLP(MLPBase):
    """
    Multilayer-perceptron network.
    """

    def __init__(self, indim, outdim, topology, activation=stax.Tanh, transform=None, seed=0):
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
class Siren(MLPBase):
    """
    Sinusoidal Representation Network as decribed in [1]

    [1] Sitzmann, V., Martel, J., Bergman, A., Lindell, D., and Wetzstein, G. "Implicit
    neural representations with periodic activation functions." Advances in Neural
    Information Processing Systems 33 (2020).
    """

    def __init__(self, indim, outdim, topology, omega_0=16, omega=1, transform=None, seed=0):
        """
        Arguments
        ---------

        indim:
            Dimensionality of the a single data input.

        outdim:
            Dimensionality of the a single data output.

        topology: Tuple
            Defines the structure of the inner layers for the network.

        omega_0:
            Weight distribution factor ω₀ for the first layer (as described in [1]).

        transform: Optional[Callable]
            Optional pre-network transformation function.

        seed: Optional[int]
            Initial seed for weight initialization.
        """

        tlayer = build_transform_layer(transform)
        # Weight initialization for the first layer of Sirens
        pdf_0 = uniform_scaling(1.0, "fan_in", scale_transform=identity)
        # Build layers
        layers_0 = [stax.Flatten, *tlayer, SirenLayer(topology[0], omega_0, pdf_0)]
        layers = [SirenLayer(i, omega) for i in topology[1:]]
        layers = layers_0 + layers + [SirenLayer(outdim, omega, is_linear=True)]
        #
        super().__init__(indim, layers, seed)


def SirenLayer(
    out_dim,
    omega,
    W_init: Optional[Callable] = None,
    b_init: Optional[Callable] = None,
    is_linear: bool = False,
):
    """
    Constructor function for a dense (fully-connected) layer for Sinusoidal Representation
    Networks. Similar to `jax.example_libraries.stax.Dense`.
    """

    if W_init is None:
        W_init = uniform_scaling(6.0 / omega**2, "fan_in")

    if b_init is None:
        b_init = uniform_scaling(1.0, "fan_in", bias_like=True)

    if is_linear:
        σ = identity(lambda x, omega: x)
    else:
        σ = identity(lambda x, omega: np.sin(omega * x))

    def init(rng, input_shape):
        k1, k2 = random.split(rng)
        inner_shape = (input_shape[-1], out_dim)
        output_shape = input_shape[:-1] + (out_dim,)
        W, b = W_init(k1, inner_shape), b_init(k2, inner_shape)
        return output_shape, (W, b)

    def apply(params, inputs, **_):
        # NOTE: experiment with having omega in params
        W, b = params
        return σ(np.dot(inputs, W) + b, omega)

    return init, apply


@dispatch
def build_transform_layer(transform: Optional[Callable] = None):
    return () if transform is None else (stax.elementwise(transform),)
