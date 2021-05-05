# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)

import jax.numpy as np

from abc import ABC, abstractproperty
from jax import grad, jit
from jax.numpy import linalg
from inspect import signature
from plum import dispatch
from typing import Callable, List, Tuple, Union
from jaxlib.xla_extension import DeviceArray as JaxArray


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
    def __init__(self, indices):
        indices, groups = process_groups(indices)
        self.indices = indices
        self.groups = groups
    #
    @abstractproperty
    def function(self):
        pass


class AxisCV(CollectiveVariable):
    """
    Similar to CollectiveVariable, but requires that an axis is provided.
    """
    def __init__(self, indices, axis):
        indices, groups = process_groups(indices)
        self.indices = indices
        self.groups = groups
        self.axis = axis
    #
    @abstractproperty
    def function(self):
        pass


class TwoPointCV(CollectiveVariable):
    """
    Similar to CollectiveVariable, but checks at initialization that only two
    indices or groups are provided.
    """
    def __init__(self, indices):
        indices, groups = process_groups(indices)
        check_groups_size(indices, groups, 2)
        self.indices = indices
        self.groups = groups
    #
    @abstractproperty
    def function(self):
        pass


class ThreePointCV(CollectiveVariable):
    """
    Similar to CollectiveVariable, but checks at initialization that only three
    indices or groups are provided.
    """
    def __init__(self, indices):
        indices, groups = process_groups(indices)
        check_groups_size(indices, groups, 3)
        self.indices = indices
        self.groups = groups
    #
    @abstractproperty
    def function(self):
        pass


class FourPointCV(CollectiveVariable):
    """
    Similar to CollectiveVariable, but checks at initialization that only four
    indices or groups are provided.
    """
    def __init__(self, indices):
        indices, groups = process_groups(indices)
        check_groups_size(indices, groups, 4)
        self.indices = indices
        self.groups = groups
    #
    @abstractproperty
    def function(self):
        pass


# ========= #
#   Utils   #
# ========= #

def check_groups_size(indices, groups, n):
    m = np.size(indices, 0) - sum(np.size(g, 0) - 1 for g in groups)
    if m != n:
        error_msg = (
            f"Exactly {n} indices or groups must be provided " +
            f"(got {m})"
        )
        raise ValueError(error_msg)


def get_nargs(f: Callable):
    return len(signature(f).parameters)


@dispatch
def build(cv: CollectiveVariable, J = grad):
    # TODO: Add support for passing weights of compute weights from masses, and
    # to reduce groups with barycenter
    ξ = cv.function
    I = cv.indices
    #
    if get_nargs(ξ) == 1:
        def evaluate(positions: JaxArray, ids: JaxArray, **kwargs):
            rs = positions[ids[I]]
            return np.asarray(ξ(rs, **kwargs))
    else:
        def evaluate(positions: JaxArray, ids: JaxArray, **kwargs):
            rs = positions[ids[I]]
            return np.asarray(ξ(*rs, **kwargs))
    #
    f, Jf = jit(evaluate), jit(J(evaluate))
    #
    def apply(positions: JaxArray, ids: JaxArray, **kwargs):
        rs = positions[:, :3]
        ξ = np.expand_dims(f(rs, ids, **kwargs).flatten(), 0)
        Jξ = np.expand_dims(Jf(rs, ids, **kwargs).flatten(), 0)
        return ξ, Jξ
    #
    return jit(apply)


@dispatch
def build(cv: CollectiveVariable, *cvs: CollectiveVariable):
    cvs = [build(cv)] + [build(cv) for cv in cvs]
    #
    def apply(positions: JaxArray, ids: JaxArray):
        ξs, Jξs = [], []
        for i in range(len(cvs)):
            ξ, Jξ = cvs[i](positions, ids)
            ξs.append(ξ)
            Jξs.append(Jξ)
        return np.hstack(ξs), np.vstack(Jξs)
    #
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


# ========== #
#   Angles   #
# ========== #

class Angle(ThreePointCV):
    @property
    def function(self):
        return angle


def angle(p1, p2, p3):
    """
    Returns the angle defined by three points in space
    (around the one in the middle).
    """
    q = p1 - p2
    r = p3 - p2
    return np.arctan2(linalg.norm(np.cross(q, r)), np.dot(q, r))


class DihedralAngle(FourPointCV):
    @property
    def function(self):
        return dihedral_angle


def dihedral_angle(p1, p2, p3, p4):
    """
    Returns the dihedral angle defined by four points in space
    (around the line defined by the two central points).
    """
    q = p3 - p2
    r = np.cross(p2 - p1, q)
    s = np.cross(q, p4 - p3)
    return np.arctan2(np.dot(np.cross(r, s), q), np.dot(r, s) * linalg.norm(q))


# ===================== #
#   Shape Descriptors   #
# ===================== #

def gyration_tensor(positions):
    n = positions.shape[0]
    S = np.zeros((3, 3))
    for r in positions:
        S += np.outer(r, r)
    return S / n


def weighted_gyration_tensor(positions, weights):
    n = positions.shape[0]
    S = np.zeros((3, 3))
    for i in range(n):
        w, r = weights[i], positions[i]
        S += w * np.outer(r, r)
    return S


class RadiusOfGyration(CollectiveVariable):
    @property
    def function(self):
        return radius_of_gyration


def radius_of_gyration(positions):
    n = positions.shape[0]
    S = np.zeros((3,))
    # TODO: Replace by `np.sum` and `vmap`
    for r in positions:
        S[:] += np.dot(r, r)
    return S / n


def weighted_radius_of_gyration(positions, weights):
    n = positions.shape[0]
    R2 = np.zeros((3,))
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        w, r = weights[i], positions[i]
        R2 += w * np.dot(r, r)
    return R2


class PrincipalMoment(AxisCV):
    @property
    def function(self):
        return (lambda rs: principal_moments(rs)[self.axis])


def principal_moments(positions):
    return linalg.eigvals(gyration_tensor(positions))


class Asphericity(CollectiveVariable):
    @property
    def function(self):
        return asphericity


def asphericity(positions):
    λ1, λ2, λ3 = principal_moments(positions)
    return λ3 - (λ1 + λ2) / 2


class Acylindricity(CollectiveVariable):
    @property
    def function(self):
        return acylindricity


def acylindricity(positions):
    λ1, λ2, _ = principal_moments(positions)
    return (λ2 - λ1)


class ShapeAnysotropy(CollectiveVariable):
    @property
    def function(self):
        return shape_anysotropy


def shape_anysotropy(positions):
    λ1, λ2, λ3 = principal_moments(positions)
    return (3 * (λ1**2 + λ2**2 + λ3**2) / (λ1 + λ2 + λ3)**2 - 1) / 2


# ======================== #
#   Particle Coordinates   #
# ======================== #

def barycenter(positions):
    return np.sum(positions, axis=0) / positions.shape[0]


def weighted_barycenter(positions, weights):
    n = positions.shape[0]
    R = np.zeros(3)
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        w, r = weights[i], positions[i]
        R += w * r
    return R


class Component(AxisCV):
    @property
    def function(self):
        return (lambda rs: barycenter(rs)[self.axis])


class Distance(TwoPointCV):
    @property
    def function(self):
        return distance


def distance(r1, r2):
    return linalg.norm(r1 - r2)
