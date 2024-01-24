# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jax import numpy as np
from jax.lax import while_loop
from jax.scipy import signal

from pysages.ml.optimizers import build
from pysages.typing import JaxArray, NamedTuple


class NNData(NamedTuple):
    params: JaxArray
    mean: JaxArray
    std: JaxArray


def normalize(data, axes=None):
    mean = data.mean(axis=axes)
    std = data.std(axis=axes)
    return (data - mean) / std, mean, std


def convolve(data, kernel, boundary="edge"):
    """
    Wrapper around `jax.scipy.signal.convolve`. It first pads the data,
    depending on the size of the kernel, and chooses a padding mode depending
    on whether the boundaries are periodic or not.
    """
    n = kernel.ndim
    if n == 1:
        padding = (kernel.size - 1) // 2
    else:
        padding = [tuple((s - 1) // 2 for _ in range(n)) for s in kernel.shape]

    def pad(slice):
        return np.pad(slice, padding, mode=boundary)

    return signal.convolve(pad(data), kernel, mode="valid")


def build_fitting_function(model, optimizer):
    """
    Returns a function that fits the model parameters to the reference data. We
    specialize on both the model and the optimizer to partially evaluate all the
    simulation-time-independent information.
    """
    initialize, keep_iterating, update = build(optimizer, model)

    def fit(params, x, y):
        state = initialize(params, x, y)
        state = while_loop(keep_iterating, update, state)
        return state

    return fit
