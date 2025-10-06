# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from copy import deepcopy
from pathlib import Path

import numpy
from jax import numpy as np
from plum import Dispatcher

from pysages.typing import Iterable, JaxArray, Scalar, Union
from pysages.utils.compat import solve_pos_def

# PySAGES main dispatcher
dispatch = Dispatcher()


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


@dispatch
def is_file(path: Path):
    return path.is_file()


@dispatch
def is_file(path: str):
    return is_file(Path(path))


def first_or_all(seq):
    """
    Returns the only element of a sequence `seq` if its length is one,
    otherwise returns `seq` itself.
    """
    return seq[0] if len(seq) == 1 else seq


@dispatch
def last(iterable, default_value=None):
    elem = default_value
    for elem in iterable:
        pass
    return elem


@dispatch
def last(a: Union[tuple, JaxArray]):
    return a[-1]


def parse_array(s, shape: tuple = (-1, 3), transpose: bool = False):
    a = np.fromstring(s, sep=" ").reshape(shape)
    return a.T if transpose else a


@dispatch(precedence=1)
def splitlines(lines: str):
    return lines.splitlines()


@dispatch
def splitlines(lines: Iterable[str]):
    return lines


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


def linear_solver(use_pinv: bool):
    """
    Returns a function that solves the linear system `A.T @ X = B` for `X`.
    When `use_pinv == True`, `numpy.linalg.pinv` is used rather than `scipy.linalg.solve`
    (this is computationally more expensive but numerically more stable).
    """
    if use_pinv:
        # This is numerically more robust
        def tsolve(A, B):
            return np.linalg.pinv(A.T) @ B

    else:
        # Another option to benchmark against is `linalg.tensorsolve(A @ A.T, A @ B)`
        def tsolve(A, B):
            return solve_pos_def(A @ A.T, A @ B)

    return tsolve
