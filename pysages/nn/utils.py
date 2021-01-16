# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import jax

import jax.numpy as np


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
    data, structure = jax.tree_flatten(params)
    shapes = [entry.shape for entry in data]
    s = (prod(shapes[-1]), -1)
    ps = np.hstack([np.reshape(entry, s) for entry in data])
    return ps, shapes, structure


def pack(structure, shapes, params):
    inds = np.cumsum(np.array([prod(s) for s in shapes]))
    part = np.split(params, inds[:-1], axis=1)
    ps = [p.reshape(s) for (p, s) in zip(part, shapes)]
    return structure.unflatten(ps)


# %% Objectives, Costs, Regularization
def v_error(model):
    _, shapes, structure = unpack(model.parameters)
    #
    def compute(p, inputs, reference):
        "Vectorized (pointwise) error"
        params = pack(structure, shapes, p)
        prediction = model.apply(params, inputs)
        return np.float32(prediction - reference)
    #
    return compute


def sum_squares(errors):
    return np.sum(errors**2, dtype=np.float64) / 2


def l2_norm(θ):
    return (θ.flatten() @ θ.flatten().T) / 2
