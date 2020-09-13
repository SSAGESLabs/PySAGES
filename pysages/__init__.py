# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import importlib

import cupy
import jax
#
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
    """Create a view of the simulation data (dynamic snapshot) for the provided backend."""
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


def bias(snapshot, state):
    """Adds the computed bias to the forces."""
    # JAX is not designed to modify the underlying memory of arrays so we need to use
    # another library. Here we use CuPy, but might as well consider NumPy + Numba
    # (we need NumPy anyway for doing this on the CPU).
    cp_forces = cupy.asarray(snapshot.forces)
    cp_bias = cupy.asarray(state.bias.block_until_ready())
    cp_forces += cp_bias
    return None


def link(backend, simulation, sampler):
    """Creates a hook that couples the simulation to the sampling method."""
    hook = backend.Hook()
    hook.initialize_from(sampler, bias)
    backend.attach(simulation, hook)
    # Return the hook to ensure it doesn't get garbage collected within the scope
    # of this function (another option is to store it in a global).
    return hook
