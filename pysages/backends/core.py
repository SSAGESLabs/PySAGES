# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import importlib
import jax

from jax import numpy as np

from ..ssages import Box, SystemView


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
        _ACTIVE_BACKEND = importlib.import_module('.hoomd', package="backends")
    else:
        raise ValueError('Invalid backend')
    #
    return _ACTIVE_BACKEND


def view(backend, simulation):
    """Create a view of the simulation data (dynamic snapshot) for the provided backend."""
    #
    if backend.get_device_type(simulation) != "gpu":
        jax.config.update("jax_platform_name", "cpu")
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


def _set_bias(backend, simulation):
    device = backend.get_device_type(simulation)
    xp = importlib.import_module("cupy" if device == "gpu" else "numpy")
    #
    def bias(snapshot, state):
        """Adds the computed bias to the forces."""
        # TODO: Check if this can be JIT compiled with numba.
        cp_forces = xp.asarray(snapshot.forces)
        cp_bias = xp.asarray(state.bias.block_until_ready())
        cp_forces += cp_bias
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
