# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC, abstractproperty
from inspect import signature
from typing import Callable, List, Tuple, Union

from jax import grad, jit
from plum import dispatch

from pysages.utils import JaxArray

import jax.numpy as np


# ================ #
#   Base Classes   #
# ================ #

UInt32 = np.uint32
Indices = Union[int, range]


class CollectiveVariable(ABC):
    """
    Abstract base class for defining collective variables

    Initialization arguments
    ------------------------
    indices : Must be a list or tuple of atoms (ints or ranges) or groups of
        atoms. A group is specified as a nested list or tuple of atoms.

    Methods
    -------
    __init__ : When defining an new collective variable, override this method
        if you need to enforce any invariant over the indices. It can
        otherwise be ommited.

    Properties
    ----------
    function : Returns an external method that implements the actual
        computation of the collective variable.
    """

    def __init__(self, indices, n = None):
        indices, groups = process_groups(indices)

        if n is not None:
            check_groups_size(indices, groups, n)

        self.indices = indices
        self.groups = groups
        self.requires_box_unwrapping = False

    @abstractproperty
    def function(self):
        pass


# NOTE: `IndexedCV` might be a better name for this
class AxisCV(CollectiveVariable):
    """
    Similar to CollectiveVariable, but requires that an axis is provided.
    """

    def __init__(self, indices, axis):
        super().__init__(indices)
        self.axis = axis

    @abstractproperty
    def function(self):
        pass


class TwoPointCV(CollectiveVariable):
    """
    Similar to CollectiveVariable, but checks at initialization that only two
    indices or groups are provided.
    """

    def __init__(self, indices):
        super().__init__(indices, 2)

    @abstractproperty
    def function(self):
        pass


class ThreePointCV(CollectiveVariable):
    """
    Similar to CollectiveVariable, but checks at initialization that only three
    indices or groups are provided.
    """

    def __init__(self, indices):
        super().__init__(indices, 3)

    @abstractproperty
    def function(self):
        pass


class FourPointCV(CollectiveVariable):
    """
    Similar to CollectiveVariable, but checks at initialization that only four
    indices or groups are provided.
    """

    def __init__(self, indices):
        super().__init__(indices, 4)

    @abstractproperty
    def function(self):
        pass


# ========= #
#   Utils   #
# ========= #

def check_groups_size(indices, groups, n):
    m = np.size(indices, 0) - sum(np.size(g, 0) - 1 for g in groups)
    if m != n:
        raise ValueError(f"Exactly {n} indices or groups must be provided (got {m})")


def get_nargs(f: Callable):
    return len(signature(f).parameters)


def _build(cv: CollectiveVariable, J = grad):
    # TODO: Add support for passing weights of compute weights from masses, and
    # to reduce groups with barycenter
    ξ = cv.function
    I = cv.indices

    if get_nargs(ξ) == 1:
        def evaluate(positions: JaxArray, ids: JaxArray, **kwargs):
            rs = positions[ids[I]]
            return np.asarray(ξ(rs, **kwargs))
    else:
        def evaluate(positions: JaxArray, ids: JaxArray, **kwargs):
            rs = positions[ids[I]]
            return np.asarray(ξ(*rs, **kwargs))

    f, Jf = jit(evaluate), jit(J(evaluate))

    def apply(rs: JaxArray, ids: JaxArray, **kwargs):
        ξ = np.expand_dims(f(rs, ids, **kwargs).flatten(), 0)
        Jξ = np.expand_dims(Jf(rs, ids, **kwargs).flatten(), 0)
        return ξ, Jξ

    return jit(apply)


def build(cv: CollectiveVariable, *cvs: CollectiveVariable):
    cvs = [_build(cv)] + [_build(cv) for cv in cvs]

    def apply(data):
        rs = data.positions[:, :3]
        ids = data.indices

        ξs, Jξs = [], []
        for i in range(len(cvs)):
            ξ, Jξ = cvs[i](rs, ids)
            ξs.append(ξ)
            Jξs.append(Jξ)

        return np.hstack(ξs), np.vstack(Jξs)

    return jit(apply)


def process_groups(indices: Union[List, Tuple]):
    n = 0
    collected = []
    groups = []
    for obj in indices:
        group_found = is_group(obj)
        if group_found:
            collected += obj
        else:
            collected.append(obj)
        s = group_size(obj)
        if group_found:
            groups.append(np.arange(n, n + s, dtype = UInt32))
        n += s
    return UInt32(np.hstack(collected)), groups


@dispatch
def is_group(indices: Indices):
    return False


@dispatch
def is_group(group: List[Indices]):
    return True


@dispatch
def is_group(obj):
    raise ValueError("Invalid indices or group: {}".format(obj))


@dispatch
def group_size(obj: int):
    return 1


@dispatch
def group_size(obj: range):
    return len(obj)


@dispatch
def group_size(obj: Union[List, Tuple]):
    return sum(group_size(o) for o in obj)
