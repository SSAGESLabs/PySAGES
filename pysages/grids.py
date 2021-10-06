# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from dataclasses import dataclass
from jax import jit
from plum import dispatch, parametric, type_parameter
from pysages.utils import JaxArray

import jax.numpy as np


class Periodicity:
    pass


class Periodic(Periodicity):
    pass


class Aperiodic(Periodicity):
    pass


class ChebyshevDistributed:
    pass


@parametric
@dataclass
class Grid:
    lower:  JaxArray
    upper:  JaxArray
    shape:  JaxArray
    size:   JaxArray

    @classmethod
    def __infer_type_parameter__(cls, *args, **kwargs):
        return Periodic if kwargs.get("periodic", False) else Aperiodic

    def __init__(self, lower, upper, shape, **kwargs):
        self.__check_init_invariants__(**kwargs)
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.shape = np.asarray(shape)
        self.size = self.upper - self.lower

    def __check_init_invariants__(self, **kwargs):
        T = type_parameter(self)
        if not (issubclass(type(T), type) and issubclass(T, Periodicity)):
            raise TypeError("Type parameter must be a subclass of Periodicity.")
        if len(kwargs) > 1 or (len(kwargs) == 1 and "periodic" not in kwargs):
            raise ValueError("Invalid keyword argument")
        periodic = kwargs.get("periodic", T is Periodic)
        if type(periodic) is not bool:
            raise TypeError("`periodic` must be a bool.")
        if (not periodic and T is Periodic) or (periodic and T is Aperiodic):
            raise ValueError("Type parameter and keyword `periodic` do not match")

    def __repr__(self):
        T = type_parameter(self)
        P = "" if T is Aperiodic else f"[{T.__name__}]"
        return f"Grid{P} ({' x '.join(map(str, self.shape))})"

    @property
    def is_periodic(self):
        return type_parameter(self) is Periodic


@dispatch
def build_indexer(grid: Grid):
    """
    Returns a function which takes a position `x` and computes the integer
    indices of the entry within the grid that contains `x`. If `x` lies outside
    the grid, the indices returned correspond to `x = grid.upper`.
    """
    def get_index(x):
        h = grid.size / grid.shape
        idx = (x.flatten() - grid.lower) // h
        idx = np.where((idx < 0) | (idx > grid.shape), grid.shape, idx)
        return (*np.flip(np.uint32(idx)),)

    return jit(get_index)


@dispatch
def build_indexer(grid: Grid[Periodic]):
    """
    Returns a function which takes a position `x` and computes the integer
    indices of the entry within the grid that contains `x`. It `x` lies outside
    the grid boundaries, the indices are wrapped around.
    """
    def get_index(x):
        h = grid.size / grid.shape
        idx = (x.flatten() - grid.lower) // h
        idx = idx % grid.shape
        return (*np.flip(np.uint32(idx)),)

    return jit(get_index)


@dispatch
def build_indexer(grid: Grid[Aperiodic], mode: ChebyshevDistributed):
    """
    Returns a function which takes a position `x` and computes the integer
    indices of the entry within the grid that contains `x`. The bins within the
    grid are 'Chebyshev distributed' along each axis. If `x` lies outside the
    grid, the indices returned correspond to `x = grid.upper`.
    """
    def get_index(x):
        x = 2 * (grid.lower - x.flatten()) / grid.size + 1
        idx = (grid.shape * np.arccos(x)) // np.pi
        idx = np.nan_to_num(idx, nan = grid.shape)
        return (*np.flip(np.uint32(idx)),)

    return jit(get_index)
