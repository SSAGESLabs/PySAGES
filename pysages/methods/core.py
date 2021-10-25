# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC, abstractmethod
from typing import Callable, Mapping

import jax
from jax import jit
from pysages.backends import ContextWrapper
from pysages.collective_variables.core import build


# ================ #
#   Base Classes   #
# ================ #
class SamplingMethod(ABC):
    def __init__(self, cvs, *args, **kwargs):
        self.snapshot_flags = set()
        self.cv = build(*cvs)
        self._raw_cvs = tuple(cvs)
        self.args = args
        self.kwargs = kwargs

    def get_snapshot_flags(self):
        flags = self.snapshot_flags
        for cv in self._raw_cvs:
            flags = flags.union(cv.snapshot_flags)
        return flags

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        """
        Returns the snapshot, and two functions: `initialize` and `update`.
        `initialize` is intended to allocate any runtime information required
        by `update`, while `update` is intended to be called after each call to
        the wrapped context's `run` method.
        """
        pass

    def run(
        self, context_generator: Callable, timesteps: int, callback: Callable = None,
        context_args: Mapping = dict(), **kwargs
    ):
        """
        Base implementation of running a single simulation/replica with a sampling method.

        Arguments
        ---------
        context_generator: Callable
            User defined function that sets up a simulation context with the backend.
            Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
            and `simtk.openmm.Simulation` for OpenMM. The function gets `context_args`
            unpacked for additional user arguments.

        timesteps: int
            Number of timesteps the simulation is running.

        callback: Optional[Callable]
            Allows for user defined actions into the simulation workflow of the method.
            `kwargs` gets passed to the backend `run` function.
        """
        context = context_generator(**context_args)
        wrapped_context = ContextWrapper(context, self, callback)
        with wrapped_context:
            wrapped_context.run(timesteps, **kwargs)


class GriddedSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, *args, **kwargs):
        super().__init__(cvs, *args, **kwargs)
        check_dims(cvs, grid)
        self.grid = grid

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


class NNSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, topology, *args, **kwargs):
        super().__init(cvs, *args, **kwargs)
        check_dims(cvs, grid)
        self.grid = grid
        self.topology = topology

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


# ========= #
#   Utils   #
# ========= #

def check_dims(cvs, grid):
    if len(cvs) != grid.shape.size:
        raise ValueError("Grid and Collective Variable dimensions must match.")


def generalize(concrete_update, backend, on_gpu, snapshot_flags = None, jit_compile = True):
    if jit_compile:
        _jit = jit
    else:
        def _jit(x): return x

    # Positions and wrapping of positions
    if snapshot_flags and backend == "hoomd" and "wrapped_positions" in snapshot_flags:
        def transform_positions(rs, img, box):
            box_array = jax.numpy.diag(box.H)
            positions_tmp = rs[:,0:3] + img * box_array
            positions = jax.numpy.concatenate((positions_tmp, rs[:,3:4]), axis=1)
            return positions

    else:
        def transform_positions(rs, img, box): return rs

    # indices
    if backend == "openmm" and on_gpu:
        def indices(ids):
            return ids.argsort()
    else:
        def indices(ids): return ids

    _update = _jit(concrete_update)
    _transform_positions = _jit(transform_positions)
    _indices = _jit(indices)

    def update(snapshot, state):
        vms = snapshot.vel_mass
        rs = _transform_positions(snapshot.positions, snapshot.images, snapshot.box)
        ids = _indices(snapshot.ids)
        #
        return _update(state, rs=rs, vms=vms, ids=ids)

    return _jit(update)
