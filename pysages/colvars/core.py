# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES
# flake8: noqa F811

"""
Abstract base classes for collective variables.
"""

from abc import ABC, abstractmethod
from inspect import signature
from typing import Callable, Sequence, Tuple, Union

from jax import grad as jax_grad, jit, numpy as np
import numpy

from pysages.utils import JaxArray, dispatch


UInt32 = np.uint32
Indices = Union[numpy.intp, numpy.int_, numpy.intc, int, range]


class CollectiveVariable(ABC):
    """
    Abstract base class for defining collective variables

    When defining a new collective variable,
    override this method if you need to enforce any invariant over the indices.
    It can otherwise be omitted.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
        Must be a list or tuple of atoms (integers or ranges) or groups of atoms.
        A group is specified as a nested list or tuple of atoms.
    group_length: int, optional
        Specify if a fixed group length is expected.
    """

    def __init__(self, indices, group_length=None):
        indices, groups = _process_groups(indices)

        if group_length is not None:
            _check_groups_size(indices, groups, group_length)

        self.indices = indices
        self.groups = groups
        self.requires_box_unwrapping = True

    @property
    @abstractmethod
    def function(self):
        """
        Returns an external method that implements the actual computation of the
        collective variable.
        """


# NOTE: `IndexedCV` might be a better name for this
class AxisCV(CollectiveVariable):
    """
    Collective variable that specifies a Cartesian Axis in addition.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
        Must be a list or tuple of atoms (integers or ranges) or groups of atoms.
        A group is specified as a nested list or tuple of atoms.
    axis: int
        Index of the Cartesian coordinate: 0 (X), 1 (Y), 2 (Z)
    group_length: int, optional
        Specify if a fixed group length is expected.
    """

    def __init__(self, indices, axis, group_length=None):
        if axis not in (0, 1, 2):
            raise RuntimeError(f"Invalid Cartesian axis {axis} index choose 0 (X), 1 (Y), 2 (Z)")
        super().__init__(indices, group_length)
        self.axis = axis


class TwoPointCV(CollectiveVariable):
    """
    Collective variable that takes group of length 2.

    Parameters
    ----------
    indices : list[tuple(int)]
       Must be a group (size 2) of atoms.
       A group is specified as a nested list or tuple of atoms.
    """

    def __init__(self, indices):
        super().__init__(indices, 2)

    @property
    @abstractmethod
    def function(self):
        pass


class ThreePointCV(CollectiveVariable):
    """
    Collective variable that takes group of length 3.

    Parameters
    ----------
    indices : list[tuple(int)]
       Must be a group (size 3) of atoms.
       A group is specified as a nested list or tuple of atoms.
    """

    def __init__(self, indices):
        super().__init__(indices, 3)

    @property
    @abstractmethod
    def function(self):
        pass


class FourPointCV(CollectiveVariable):
    """
    Collective variable that takes group of length 4.

    Parameters
    ----------
    indices : list[tuple(int)]
       Must be a group (size 4) of atoms.
       A group is specified as a nested list or tuple of atoms.
    """

    def __init__(self, indices):
        super().__init__(indices, 4)

    @property
    @abstractmethod
    def function(self):
        pass


# ========= #
#   Utils   #
# ========= #


def _check_groups_size(indices, groups, group_length):
    input_length = np.size(indices, 0) - sum(np.size(g, 0) - 1 for g in groups)
    if input_length != group_length:
        raise ValueError(
            f"Exactly {group_length} indices or groups must be provided (got {input_length})"
        )


def _get_nargs(function: Callable):
    return len(signature(function).parameters)


def _build(cv: CollectiveVariable, grad=jax_grad):
    # TODO: Add support for compute weights from masses # pylint:disable=fixme
    xi = cv.function
    idx = cv.indices
    gps = cv.groups

    if _get_nargs(xi) == 1:

        def evaluate(positions: JaxArray, ids: JaxArray, **kwargs):
            pos = positions[ids[idx]]
            return np.asarray(xi(pos, **kwargs))

    elif len(gps) == 0:

        def evaluate(positions: JaxArray, ids: JaxArray, **kwargs):
            pos = positions[ids[idx]]
            return np.asarray(xi(*pos, **kwargs))

    else:

        def evaluate(positions: JaxArray, ids: JaxArray, **kwargs):
            pos = positions[ids[idx]]
            pos = [pos[g] for g in gps]
            return np.asarray(xi(*pos, **kwargs))

    function = jit(evaluate)

    if grad is None:

        def apply(pos: JaxArray, ids: JaxArray, **kwargs):
            return np.expand_dims(function(pos, ids, **kwargs).flatten(), 0)

    else:

        gradient = jit(grad(evaluate))

        def apply(pos: JaxArray, ids: JaxArray, **kwargs):
            xi = np.expand_dims(function(pos, ids, **kwargs).flatten(), 0)
            Jxi = np.expand_dims(gradient(pos, ids, **kwargs).flatten(), 0)
            return xi, Jxi

    return jit(apply)


def build(cv: CollectiveVariable, *cvs: CollectiveVariable, grad=jax_grad):
    """
    Jit compile and stack collective variables.

    Parameters
    ----------
    cv: CollectiveVariable
        Collective Variable object to jit compile.
    cvs: list[CollectiveVariable]
        Sequence of Collective variables that get stacked on top of each other.
    grad: Optional[Callable]
        Jax transform that is used to compute the gradient, if `None` is
        provided, only the collective variables will be computed.
        Defaults to `jax.grad`.

    Returns
    -------
    Callable[Snapshot]
        Jit compiled function, that takes a snapshot object, and access its
        positions and indices. With this information the collective variables
        (and their gradients if `grad is not None`) are computed and returned.
    """
    cvs = [_build(cv, grad=grad)] + [_build(cv, grad=grad) for cv in cvs]

    if grad is None:

        def apply(data):
            pos = data.positions[:, :3]
            ids = data.indices
            return np.hstack([cv(pos, ids) for cv in cvs])

    else:

        def apply(data):
            pos = data.positions[:, :3]
            ids = data.indices

            xis, xis_grads = [], []
            for cv in cvs:
                xi, Jxi = cv(pos, ids)
                xis.append(xi)
                xis_grads.append(Jxi)

            return np.hstack(xis), np.vstack(xis_grads)

    return jit(apply)


def _process_groups(indices: Union[Sequence, Tuple]):
    total_group_length = 0
    collected = []
    groups = []

    for obj in indices:
        group_found = _is_group(obj)
        if group_found:
            collected += obj
        else:
            collected.append(obj)
        group_length = _group_size(obj)
        if group_found:
            groups.append(
                np.arange(total_group_length, total_group_length + group_length, dtype=UInt32)
            )
        total_group_length += group_length

    return UInt32(np.hstack(collected)), groups


@dispatch
def _is_group(indices: Indices):  # pylint:disable=unused-argument
    return False


@dispatch
def _is_group(group: Sequence[Indices]):  # pylint:disable=unused-argument
    return True


@dispatch
def _is_group(obj):
    raise ValueError(f"Invalid indices or group: {obj}")


@dispatch
def _group_size(
    obj: Union[numpy.intp, numpy.intc, numpy.int_, int]
):  # pylint:disable=unused-argument
    return 1


@dispatch
def _group_size(obj: range):
    return len(obj)


@dispatch
def _group_size(obj: Union[Sequence, Tuple]):
    return sum(_group_size(o) for o in obj)
