# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jaxlib.xla_extension import DeviceArray, PyTreeDef
from typing import NamedTuple

import jax
import jax.numpy as np


class ParametersLayout(NamedTuple):
    """
    Holds the information needed to pack flatten parameters of a
    `jax.example_libraries.stax.serial` model.
    """
    structure:  PyTreeDef
    shapes:     list
    separators: DeviceArray


def rng_key(seed=0, n=2):
    """
    Returns a pseudo-randomly generated key, constructed by calling
    `jax.random.PRNGKey(seed)` and then splitting it `n` times.
    """
    key = jax.random.PRNGKey(seed)
    for _ in range(n):
        key, _ = jax.random.split(key)
    return key


def prod(xs):
    y = 1
    for x in xs:
        y *= x
    return y


# %% Models
def unpack(params):
    """
    Returns the parameters of a `jax.example_libraries.stax.serial` model stacked
    into a flat vector. This representation is more convenient for computing
    the jacobian of the errors of the model.
    """
    data, structure = jax.tree_flatten(params)
    ps = np.hstack([values.flatten() for values in data])
    shapes = [values.shape for values in data]
    separators = np.cumsum(np.array([prod(s) for s in shapes[:-1]]))
    return ps, ParametersLayout(structure, shapes, separators)


def pack(params, layout):
    """
    Repacks the flatten parameters of a `jax.example_libraries.stax.serial` model
    previously flatten with `unpack`.
    """
    structure, shapes, separators = layout
    partition = params.split(separators)
    ps = [p.reshape(s) for (p, s) in zip(partition, shapes)]
    return structure.unflatten(ps)


# %% Objectives, Costs, Regularization
def v_error(model):
    _, layout = unpack(model.parameters)

    def compute(ps, inputs, reference):
        "Vectorized (pointwise) error"
        params = pack(ps, layout)
        prediction = model.apply(params, inputs)
        return np.float32(prediction - reference)

    return compute


def sum_squares(errors):
    return np.sum(errors**2, dtype=np.float64) / 2


def l2_norm(θ):
    return (θ.flatten() @ θ.flatten().T) / 2
