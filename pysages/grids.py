# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta


from collections import namedtuple
from .utils import register_pytree_namedtuple


GridInfo = namedtuple("GridInfo", ["shape", "lower", "upper", "periodicity"])


@register_pytree_namedtuple
class Grid(GridInfo):
    def __new__(cls, shape, lower, upper, periodicity):
        if not len(shape) == len(lower) == len(upper) == len(periodicity):
            raise ValueError("All arguments must be of the same lenght.")
        return super(Grid, cls).__new__(cls, shape, lower, upper, periodicity)
    #
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)
