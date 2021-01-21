# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import importlib
import jax

from jax import numpy as np
from pysages.ssages import Box, SystemView


# Set default floating point type for arrays in `jax` to `jax.f64`
jax.config.update("jax_enable_x64", True)


# Records the backend selected with `set_backend`
_ACTIVE_BACKEND = None


def active_backend():
    return _ACTIVE_BACKEND


def supported_backends():
    return ('hoomd',)


def set_backend(name):
    """To see a list of possible backends run `supported_backends()`."""
    #
    global _ACTIVE_BACKEND
    #
    if name in supported_backends():
        _ACTIVE_BACKEND = importlib.import_module('.hoomd', package="pysages.backends")
    else:
        raise ValueError('Invalid backend')
    #
    return _ACTIVE_BACKEND


def view(backend, simulation):
    """Create a view of the simulation data (dynamic snapshot) for the provided backend."""
    #
    if not backend.is_on_gpu(simulation):
        jax.config.update("jax_platform_name", "cpu")
    #
    positions, momenta, forces, tags, H, origin, dt = backend.view(simulation)
    box = Box(np.asarray(H), np.asarray(origin))
    #
    return SystemView(positions, momenta, forces, tags, box, dt)


def _set_bias(backend, simulation):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if backend.is_on_gpu(simulation):
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


def link(backend, simulation, sampler):
    """Creates a hook that couples the simulation to the sampling method."""
    hook = backend.Hook()
    bias = _set_bias(backend, simulation)
    hook.initialize_from(sampler, bias)
    backend.attach(simulation, hook)
    # Return the hook to ensure it doesn't get garbage collected within the scope
    # of this function (another option is to store it in a global).
    return hook
