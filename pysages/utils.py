# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from copy import deepcopy
from importlib import import_module
from typing import Union

import jaxlib
import jaxlib.xla_extension as xe
import numpy
from jax import numpy as np
from jax.scipy import linalg
from jax.tree_util import register_pytree_node
from plum import Dispatcher

# PySAGES main dispatcher
dispatch = Dispatcher()


JaxArray = xe.Array
Bool = Union[JaxArray, bool]
Float = Union[JaxArray, float]
Int = Union[JaxArray, int]
Scalar = Union[None, bool, int, float]


class ToCPU:
    pass


# - https://github.com/google/jax/issues/446
# - https://github.com/google/jax/issues/806
def register_pytree_namedtuple(cls):
    register_pytree_node(
        cls,
        lambda xs: (tuple(xs), None),  # tell JAX how to unpack
        lambda _, xs: cls(*xs),  # tell JAX how to pack back
    )
    return cls


@dispatch
def copy(x: Scalar):
    return x


@dispatch(precedence=1)
def copy(t: tuple, *args):
    return tuple(copy(x, *args) for x in t)


@dispatch
def copy(x: JaxArray):
    return x.copy()


@dispatch
def copy(x, _: ToCPU):
    return deepcopy(x)


@dispatch
def copy(x: JaxArray, _: ToCPU):
    return numpy.asarray(x._value)


def identity(x):
    return x


def row_sum(x):
    """
    Sum array `x` along each of its row (`axis = 1`),
    """
    return np.sum(x.reshape(np.size(x, 0), -1), axis=1)


def gaussian(a, sigma, x):
    """
    N-dimensional origin-centered gaussian with height `a` and standard deviation `sigma`.
    """
    return a * np.exp(-row_sum((x / sigma) ** 2) / 2)


# Compatibility utils


def try_import(new_name, old_name):
    try:
        return import_module(new_name)
    except ModuleNotFoundError:
        return import_module(old_name)


def _version_as_tuple(ver_str):
    return tuple(int(i) for i in ver_str.split(".") if i.isdigit())


if _version_as_tuple(jaxlib.__version__) < (0, 3, 15):

    def solve_pos_def(a, b):
        return linalg.solve(a, b, sym_pos="sym")

else:

    def solve_pos_def(a, b):
        return linalg.solve(a, b, assume_a="pos")
