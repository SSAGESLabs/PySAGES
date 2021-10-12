# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC, abstractmethod
from typing import Callable

from jax import jit
from pysages.backends import ContextWrapper
from pysages.collective_variables.core import build


# ================ #
#   Base Classes   #
# ================ #
class SamplingMethod(ABC):
    def __init__(self, cvs, *args, **kwargs):
        self.cv = build(*cvs)
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def build(self, *args, **kwargs):
        """
        Build the method for JAX execution.
        """
        pass

    def run(self, context_generator: Callable, timesteps: int, callback: Callable, context_args=dict(), **kwargs):
        """
        Base implementation of running a single simulation/replica with a sampling method.

        context_generator: user defined function that sets up a simulation context with the backend.
                           Must return an instance of hoomd.conext.SimulationContext for hoomd-blue and simtk.openmm.openmm.Context.
                           The function gets context_args unpacked for additional user args.
        timesteps: number of timesteps the simulation is running.
        callback: Callback to integrate user defined actions into the simulation workflow of the method
        kwargs gets passed to the backend run function for additional user arguments to be passed down.
        """
        context = context_generator()
        wrapped_context = ContextWrapper(context, self, callback)
        with wrapped_context:
            wrapped_context.run(timesteps, **kwargs)


class GriddedSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, *args, **kwargs):
        check_dims(cvs, grid)
        self.cv = build(*cvs)
        self.grid = grid
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def build(self, *args, **kwargs):
        pass


class NNSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, topology, *args, **kwargs):
        check_dims(cvs, grid)
        self.cv = build(*cvs)
        self.grid = grid
        self.topology = topology
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def build(self, *args, **kwargs):
        pass


# ========= #
#   Utils   #
# ========= #

def check_dims(cvs, grid):
    if len(cvs) != grid.shape.size:
        raise ValueError("Grid and Collective Variable dimensions must match.")


def generalize(concrete_update, jit_compile = True):
    if jit_compile:
        _jit = jit
    else:
        def _jit(x): return x

    _update = _jit(concrete_update)

    def update(snapshot, state):
        vms = snapshot.vel_mass
        rs = snapshot.positions
        ids = snapshot.ids
        #
        return _update(state, rs, vms, ids)

    return _jit(update)
