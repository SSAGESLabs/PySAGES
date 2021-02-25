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
    check_backend_initialization()
    #
    if not _CURRENT_BACKEND.is_on_gpu(simulation):
        jax.config.update("jax_platform_name", "cpu")
    #
    positions, momenta, forces, tags, H, origin, dt = _CURRENT_BACKEND.view(simulation)
    box = Box(np.asarray(H), np.asarray(origin))
    #
    return SystemView(positions, momenta, forces, tags, box, dt)


def bind(simulation, sampler):
    """Couples the sampling method to the simulation."""
    #
    check_backend_initialization()
    #
    return _CURRENT_BACKEND.bind(simulation, sampler)


def check_backend_initialization():
    if _CURRENT_BACKEND is None:
        raise RuntimeError("No backend has been set")
