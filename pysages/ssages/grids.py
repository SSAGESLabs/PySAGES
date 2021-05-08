# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


from collections import namedtuple

import jax.numpy as np
from jax import jit
from jax.lax import convert_element_type
from pysages.utils import register_pytree_namedtuple


GridInfo = namedtuple("GridInfo", ["lower", "upper", "shape", "periodicity", "Δ"])


@register_pytree_namedtuple
class Grid(GridInfo):
    def __new__(cls, lower, upper, shape, periodicity, Δ = None):
        if not len(shape) == len(lower) == len(upper) == len(periodicity):
            raise ValueError("All arguments must be of the same lenght.")
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        shape = np.asarray(shape)
        periodicity = np.asarray(periodicity)
        Δ = np.divide(upper - lower, shape)
        return super(Grid, cls).__new__(cls, lower, upper, shape, periodicity, Δ)
    #
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


@jit
def get_index(grid, x):
    I = np.floor_divide(x - grid.lower, grid.Δ).flatten()
    return (*np.uint32(I),)
