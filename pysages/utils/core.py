# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from copy import deepcopy
from typing import Union

import numpy
from jax import numpy as np
from plum import Dispatcher

from pysages.utils.compat import JaxArray

# PySAGES main dispatcher
dispatch = Dispatcher()

Bool = Union[JaxArray, bool]
Float = Union[JaxArray, float]
Int = Union[JaxArray, int]
Scalar = Union[None, bool, int, float]


class ToCPU:
    pass


@dispatch
def copy(x: Scalar):
    return x


@dispatch(precedence=1)
def copy(t: tuple, *args):  # noqa: F811 # pylint: disable=C0116,E0102
    return tuple(copy(x, *args) for x in t)  # pylint: disable=E1120


@dispatch
def copy(x: JaxArray):  # noqa: F811 # pylint: disable=C0116,E0102
    return x.copy()


@dispatch
def copy(x, _: ToCPU):  # noqa: F811 # pylint: disable=C0116,E0102
    return deepcopy(x)


@dispatch
def copy(x: JaxArray, _: ToCPU):  # noqa: F811 # pylint: disable=C0116,E0102
    return numpy.asarray(x._value)  # pylint: disable=W0212


def identity(x):
    return x


def eps(T: type = np.zeros(0).dtype):
    return np.finfo(T).eps


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
