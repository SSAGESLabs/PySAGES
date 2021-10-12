# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC, abstractmethod

from jax import jit
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
