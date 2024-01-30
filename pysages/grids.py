# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from dataclasses import dataclass

from jax import jit
from jax import numpy as np
from plum import Union, parametric

from pysages.typing import JaxArray
from pysages.utils import dispatch, is_generic_subclass, prod


class GridType:
    pass


class Periodic(GridType):
    pass


class Regular(GridType):
    pass


class Chebyshev(GridType):
    pass


@parametric
@dataclass
class Grid:
    lower: JaxArray
    upper: JaxArray
    shape: JaxArray
    size: JaxArray

    @classmethod
    def __infer_type_parameter__(cls, *_, **kwargs):
        return Periodic if kwargs.get("periodic", False) else Regular

    def __init__(self, lower, upper, shape, **kwargs):
        self.__check_init_invariants__(**kwargs)
        shape = np.asarray(shape)
        n = shape.size
        self.lower = np.asarray(lower).reshape(n)
        self.upper = np.asarray(upper).reshape(n)
        self.shape = shape.reshape(n)
        self.size = self.upper - self.lower

    def __check_init_invariants__(self, **kwargs):
        T = type(self).type_parameter  # pylint: disable=E1101
        if not (issubclass(type(T), type) and issubclass(T, GridType)):
            raise TypeError("Type parameter must be a subclass of GridType.")
        if len(kwargs) > 1 or (len(kwargs) == 1 and "periodic" not in kwargs):
            raise ValueError("Invalid keyword argument")
        periodic = kwargs.get("periodic", T is Periodic)
        if type(periodic) is not bool:
            raise TypeError("`periodic` must be a bool.")
        type_kw_mismatch = (not periodic and T is Periodic) or (
            periodic and is_generic_subclass(Union[T], Union[Regular, Chebyshev])
        )
        if type_kw_mismatch:
            raise ValueError("Incompatible type parameter and keyword argument")

    def __repr__(self):
        T = type(self).type_parameter  # pylint: disable=E1101
        P = "" if T is Regular else f"[{T.__name__}]"
        return f"Grid{P} ({' x '.join(map(str, self.shape))})"

    @property
    def is_periodic(self):
        return type(self).type_parameter is Periodic  # pylint: disable=E1101


@dispatch
def build_grid(T, lower, upper, shape):
    return Grid[T](lower, upper, shape)


@dispatch
def build_grid(grid: type(None)):  # noqa: F811 # pylint: disable=C0116,E0102
    return grid


@dispatch
def convert(grid: Grid, T: type):
    if not issubclass(T, Grid):
        raise TypeError(f"Cannot convert Grid to a {repr(T)}")
    return T(grid.lower, grid.upper, grid.shape)


@dispatch
def get_info(grid: Grid):
    T = type(grid).type_parameter
    grid_args = (
        tuple(float(x) for x in grid.lower),
        tuple(float(x) for x in grid.upper),
        tuple(int(x) for x in grid.shape),
    )
    return (T, *grid_args)


@dispatch
def get_info(grid: type(None)):  # noqa: F811 # pylint: disable=C0116,E0102
    return (grid,)


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
        return (*np.uint32(idx),)

    return jit(get_index)


@dispatch
def build_indexer(grid: Grid[Periodic]):  # noqa: F811 # pylint: disable=C0116,E0102
    """
    Returns a function which takes a position `x` and computes the integer
    indices of the entry within the grid that contains `x`. It `x` lies outside
    the grid boundaries, the indices are wrapped around.
    """

    def get_index(x):
        h = grid.size / grid.shape
        idx = (x.flatten() - grid.lower) // h
        idx = idx % grid.shape
        return (*np.uint32(idx),)

    return jit(get_index)


@dispatch
def build_indexer(grid: Grid[Chebyshev]):  # noqa: F811 # pylint: disable=C0116,E0102
    """
    Returns a function which takes a position `x` and computes the integer
    indices of the entry within the grid that contains `x`. The bins within the
    grid are 'Chebyshev distributed' along each axis. If `x` lies outside the
    grid, the indices returned correspond to `x = grid.upper`.
    """

    def get_index(x):
        x = 2 * (grid.lower - x.flatten()) / grid.size + 1
        idx = (grid.shape * np.arccos(x)) // np.pi
        idx = np.nan_to_num(idx, nan=grid.shape)
        return (*np.uint32(idx),)

    return jit(get_index)


def grid_transposer(grid):
    """
    Returns a function that transposes arrays mapped to a `Grid`.

    The result function takes an array, reshapes it to match the grid dimensions,
    transposes it along the first axes. The first axes are assumed to correspond to the
    axes of the grid.
    """
    d = len(grid.shape)
    shape = (*grid.shape,)
    axes = (*reversed(range(d)),)
    n = grid.shape.prod().item()

    def transpose(array: JaxArray):
        m = prod(array.shape) // n
        return array.reshape(*shape, m).transpose(*axes, d).squeeze()

    return transpose
