# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta


import importlib

import cupy
import jax
from jax import numpy as np
from jax.dlpack import from_dlpack
#
jax.config.update("jax_enable_x64", True)

from . import cvs
from . import methods
from .cvs import collective_variable
from .grids import Grid
from .snapshot import Box, SystemView


# Visible exports
__all__ = ["active_backend", "link", "set_backend", "supported_backends", "view"]


# Records the backend selected with `set_backend`
_ACTIVE_BACKEND = None
#
def active_backend(): return _ACTIVE_BACKEND
def supported_backends(): return ('hoomd',)


def set_backend(name):
    """To see a list of possible backends run `supported_backends()`."""
    #
    global _ACTIVE_BACKEND
    #
    if name in supported_backends():
        _ACTIVE_BACKEND = importlib.import_module(name + '.dlext')
    else:
        raise ValueError('Invalid backend')
    #
    return _ACTIVE_BACKEND


def view(backend, simulation):
    """Create a view of the simulation data from the active backend."""
    #
    convert = jax.dlpack.from_dlpack
    #
    R, V_M, F, I, B, dt = backend.view(simulation)
    positions = convert(R)
    vel_mass = convert(V_M)
    forces = convert(F)
    tags = convert(I)
    box = Box(np.asarray(B[0]), np.asarray(B[1]))
    #
    return SystemView(positions, vel_mass, forces, tags, box, dt)


def bias(state):
    cp_forces = cupy.asarray(state.snapshot.forces)
    cp_bias = cupy.asarray(state.bias)
    cp_forces += cp_bias
    return None


def link(backend, simulation, sampler):
    hook = backend.Hook()
    hook.initialize_from(sampler, bias)
    backend.attach(simulation, hook)
    # Return the hook to ensure it doesn't get garbage collected
    # within the scope of this function
    return hook
