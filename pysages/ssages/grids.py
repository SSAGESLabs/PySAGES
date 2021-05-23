# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from collections import namedtuple
from jax import jit

import jax.numpy as np


class Grid(namedtuple(
    "Grid",
    (
        "lower",
        "size",
        "shape",
        "is_periodic",
    )
)):
    def __new__(cls, lower, upper, shape, periodic):
        if not len(shape) == len(lower) == len(upper):
            raise ValueError("All arguments must be of the same lenght.")
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        shape = np.asarray(shape)
        size = (upper - lower)
        return super().__new__(cls, lower, size, shape, periodic)

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


def build_indexer(grid):
    if grid.is_periodic:
        def get_grid_index(x):
            h = grid.size / grid.shape
            idx = np.floor_divide(x - grid.lower, h).flatten()
            return (*np.flip(np.uint32(idx)),)
    else:
        def get_grid_index(x):
            x = 2 * (grid.lower - x) / grid.size + 1
            idx = np.floor_divide(grid.shape * np.arccos(x), np.pi).flatten()
            return (*np.flip(np.uint32(idx)),)

    return jit(get_grid_index)
