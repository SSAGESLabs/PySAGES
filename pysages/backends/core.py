# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import importlib
import jax
import warnings

from jax import numpy as np
from pysages.ssages import Box, SystemView


# Set default floating point type for arrays in `jax` to `jax.f64`
jax.config.update("jax_enable_x64", True)


# Records the backend selected with `set_backend`
_CURRENT_BACKEND = None


def current_backend():
    if _CURRENT_BACKEND is not None:
        return _CURRENT_BACKEND
    warnings.warn("No backend has been set")


def supported_backends():
    return ('hoomd',)


def set_backend(name):
    """To see a list of possible backends run `supported_backends()`."""
    #
    global _CURRENT_BACKEND
    #
    if name in supported_backends():
        _CURRENT_BACKEND = importlib.import_module('.' + name, package="pysages.backends")
    else:
        raise ValueError("Invalid backend")
    #
    return _CURRENT_BACKEND


def wrap(simulation):
    """Create a view of the simulation data (dynamic snapshot) for the provided backend."""
    #
    if _CURRENT_BACKEND is None:
        raise RuntimeError("No backend has been set")
    #
    if not _CURRENT_BACKEND.is_on_gpu(simulation):
        jax.config.update("jax_platform_name", "cpu")
    #
    positions, momenta, forces, tags, H, origin, dt = _CURRENT_BACKEND.view(simulation)
    box = Box(np.asarray(H), np.asarray(origin))
    #
    return SystemView(positions, momenta, forces, tags, box, dt)


def _set_bias(simulation):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if _CURRENT_BACKEND.is_on_gpu(simulation):
        cupy = importlib.import_module("cupy")
        wrap = cupy.asarray
    else:
        utils = importlib.import_module(".utils", package="pysages.backends")
        wrap = utils.view
    #
    def bias(snapshot, state):
        """Adds the computed bias to the forces."""
        # TODO: Factor out the views so we can eliminate two function calls here.
        # Also, check if this can be JIT compiled with numba.
        forces = wrap(snapshot.forces)
        biases = wrap(state.bias.block_until_ready())
        forces += biases
        return None
    #
    return bias


def bind(simulation, sampler):
    """Creates a hook that couples the simulation to the sampling method."""
    #
    if _CURRENT_BACKEND is None:
        raise RuntimeError("No backend has been set")
    #
    hook = _CURRENT_BACKEND.Hook()
    bias = _set_bias(simulation)
    hook.initialize_from(sampler, bias)
    _CURRENT_BACKEND.attach(simulation, hook)
    # Return the hook to ensure it doesn't get garbage collected within the scope
    # of this function (another option is to store it in a global).
    return hook
