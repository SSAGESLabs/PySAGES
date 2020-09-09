# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


from collections import namedtuple
from functools import partial

from jax import numpy as np
from jax import grad, jacfwd, jacrev, jit, value_and_grad, vmap
from jax.numpy import linalg


# Closures that capture the indices of a collective variable and returns the both the
# reaction coordinate function as well as its gradient or Jacobian, which take as
# arguments the positions and tags of all particles.
#
def value_and_jac_over(indices, J = grad, ξ = None):
    # For reaction coordinates functions that take a fixed number of particle positions
    # (e.g. Angle, DihedralAngle).
    #
    if ξ is None:
        return partial(value_and_jac_over, indices, J)
    #
    ξ = jit(ξ)
    def wrapper(positions, tags, **kwargs):
        ps = positions[tags.argsort()[indices], 0:3]
        return np.asarray(ξ(*ps, **kwargs))
    #
    return jit(wrapper), jit(J(wrapper))
#
def value_and_jac_over_array(indices, J = grad, ξ = None):
    # For reaction_coordinates that take an array of positions
    # (e.g., RadiusOfGyration, Barycenter).
    #
    if ξ is None:
        return partial(value_and_jac_over_array, indices, J)
    #
    ξ = jit(ξ)
    def wrapper(positions, tags, **kwargs):
        ps = positions[tags.argsort()[indices], 0:3]
        return np.asarray(ξ(ps, **kwargs))
    #
    return jit(wrapper), jit(J(wrapper))


CollectiveVariableData = namedtuple("CollectiveVariableData", ["dims", "ξ", "Jξ"])


def collective_variable(cv0, *cvs):
    n0, ξ0, Jξ0 = cv0
    if len(cvs) >= 1:
        n1, ξ1, Jξ1 = cvs[0]
    if len(cvs) == 2:
        n2, ξ2, Jξ2 = cvs[1]
    #
    if len(cvs) == 0 and n0 <= 3:
        def wrapper(positions, tags):
            return ξ0(positions, tags).flatten(), Jξ0(positions, tags).flatten()
        return CollectiveVariableData(n0, jit(wrapper), None)
    #
    if len(cvs) == 1 and n0 + n1 <= 3:
        def wrapper(positions, tags):
            ξs = np.stack((
                ξ0(positions, tags).flatten(),
                ξ1(positions, tags).flatten()
            ))
            Jξs = np.vstack((
                Jξ0(positions, tags).flatten(),
                Jξ1(positions, tags).flatten()
            ))
            return ξs, Jξs
        return CollectiveVariableData(n0 + n1, jit(wrapper), None)
    #
    if len(cvs) == 2 and n0 + n1 + n2 == 3:
        def wrapper(positions, tags):
            ξs = np.stack((
                ξ0(positions, tags),
                ξ1(positions, tags),
                ξ2(positions, tags)
            ))
            Jξs = np.vstack((
                Jξ0(positions, tags).flatten(),
                Jξ1(positions, tags).flatten(),
                Jξ2(positions, tags).flatten()
            ))
            return ξs, Jξs
        return CollectiveVariableData(n0 + n1 + n2, jit(wrapper), None)
    #
    raise ValueError(
        "Collective variable spaces of more than "
        "three dimensions currently unsupported."
    )


#==========#
#  Angles  #
#==========#

def _angle(p1, p2, p3):
    q = p1 - p2
    r = p3 - p2
    return np.arctan2(linalg.norm(np.cross(q, r)), np.dot(q, r))


def angle(indices):
    """
    Returns a function that computes the angle defined by three points in space
    (specified by `indices`) around the one in the middle.
    """
    if np.size(indices, 0) != 3:
        raise ValueError('Exactly three indices must be provided (got {indices.size}).')
    #
    ξ, Jξ = value_and_jac_over(indices)(_angle)
    return CollectiveVariableData(1, jit(ξ), jit(Jξ))


def _dihedral_angle(p1, p2, p3, p4):
    q = p3 - p2
    r = np.cross(p2 - p1, q)
    s = np.cross(q, p4 - p3)
    return np.arctan2(np.dot(np.cross(r, s), q), np.dot(r, s) * linalg.norm(q))


def dihedral_angle(indices):
    """
    Returns a function that computes the dihedral angle, as well as its gradient, defined
    by four points in space (around the line defined by the two central points).
    """
    if np.size(indices, 0) != 4:
        raise ValueError('Exactly four indices must be provided (got {indices.size}).')
    #
    ξ, Jξ = value_and_jac_over(indices)(_dihedral_angle)
    return CollectiveVariableData(1, jit(ξ), jit(Jξ))


#=====================#
#  Shape Descriptors  #
#=====================#

def _gyration_tensor(positions):
    n = positions.shape[0]
    S = np.zeros((3, 3))
    for r in positions:
        S += np.outer(r, r)
    return S / n


def _weighted_gyration_tensor(positions, weights):
    n = positions.shape[0]
    S = np.zeros((3, 3))
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        w, r = weights[i], positions[i]
        S += w * np.outer(r, r)
    return S


def gyration_tensor(indices, weights = None):
    """
    Gyration tensor of group of particles. If weights are supplied, the weighted average
    will be computed.
    """
    if weights is None:
        reaction_coordinate = _gyration_tensor
    else:
        ws = weights / np.sum(weights)
        reaction_coordinate = lambda rs : _weighted_gyration_tensor(rs, ws)
    #
    J = jacfwd if np.size(indices, 0) <= 3 else jacrev
    ξ, Jξ = value_and_jac_over_array(indices, J)(reaction_coordinate)
    return CollectiveVariableData(9, jit(ξ), jit(Jξ))


def _radius_of_gyration(positions):
    n = positions.shape[0]
    S = np.zeros((3,))
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        S[:] += np.dot(r, r)
    return S / n


def _weighted_radius_of_gyration(positions, weights):
    n = positions.shape[0]
    R2 = np.zeros((3,))
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        w, r = weights[i], positions[i]
        R2 += w * np.dot(r, r)
    return R2


def radius_of_gyration(indices, weights = None):
    if weights is None:
        reaction_coordinate = _radius_of_gyration
    else:
        ws = weights / np.sum(weights)
        def reaction_coordinate(rs):
            return _weighted_radius_of_gyration(rs, ws)
    #
    ξ, Jξ = value_and_jac_over_array(indices)(reaction_coordinate)
    return CollectiveVariableData(1, jit(ξ), jit(Jξ))


_principal_moments = linalg.eigvals


def principal_moments(indices, weights = None):
    if weights is None:
        def reaction_coordinate(rs):
            return _principal_moments(_gyration_tensor(rs))
    else:
        ws = weights / np.sum(weights)
        def reaction_coordinate(rs):
            return _principal_moments(_weighted_gyration_tensor(rs, ws))
    #
    ξ, Jξ = value_and_jac_over_array(indices, jacrev)(reaction_coordinate)
    return CollectiveVariableData(3, jit(ξ), jit(Jξ))


def asphericity(indices, weights = None):
    if weights is None:
        def reaction_coordinate(rs):
            return _principal_moments(_gyration_tensor(rs))
    else:
        ws = weights / np.sum(weights)
        def reaction_coordinate(rs):
            return _principal_moments(_weighted_gyration_tensor(rs, ws))
    #
    def asphericity(positions):
        λ1, λ2, λ3 = reaction_coordinate(positions)
        return λ3 - (λ1 + λ2) / 2
    #
    ξ, Jξ = value_and_jac_over_array(indices)(asphericity)
    return CollectiveVariableData(1, jit(ξ), jit(Jξ))


def acylindricity(indices, weights = None):
    if weights is None:
        def reaction_coordinate(rs):
            return _principal_moments(_gyration_tensor(rs))
    else:
        ws = weights / np.sum(weights)
        def reaction_coordinate(rs):
            return _principal_moments(_weighted_gyration_tensor(rs, ws))
    #
    def acylindricity(positions):
        λ1, λ2, λ3 = reaction_coordinate(positions)
        return (λ2 - λ1)
    #
    ξ, Jξ = value_and_jac_over_array(indices)(acylindricity)
    return CollectiveVariableData(1, jit(ξ), jit(Jξ))


def shape_anysotropy(indices, weights = None):
    if weights is None:
        def reaction_coordinate(rs):
            return _principal_moments(_gyration_tensor(rs))
    else:
        ws = weights / np.sum(weights)
        def reaction_coordinate(rs):
            return _principal_moments(_weighted_gyration_tensor(rs, ws))
    #
    def shape_anysotropy(positions):
        λ1, λ2, λ3 = reaction_coordinate(positions)
        return (3 * (λ1**2 + λ2**2 + λ3**2) / (λ1 + λ2 + λ3)**2 - 1) / 2
    #
    ξ, Jξ = value_and_jac_over(indices)(shape_anysotropy)
    return CollectiveVariableData(1, jit(ξ), jit(Jξ))


#========================#
#  Particle Coordinates  #
#========================#

def _barycenter(positions):
    return np.sum(positions, axis=0) / positions.shape[0]


def _weighted_barycenter(positions, weights):
    n = positions.shape[0]
    R = np.zeros(3)
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        w, r = weights[i], positions[i]
        R += w * r
    return R


def barycenter(indices, weights = None):
    if weights is None:
        reaction_coordinate = _barycenter
    else:
        ws = weights / sum(weights)
        def reaction_coordinate(rs):
            return _weighted_barycenter(rs, ws)
    #
    ξ, Jξ = value_and_jac_over(indices)(_barycenter)
    return CollectiveVariableData(3, jit(ξ), jit(Jξ))


def _barycenter_component(positions, axis):
    return np.sum(positions[:, axis]) / positions.shape[0]


def _weighted_barycenter_component(positions, axis, weights):
    n = positions.shape[0]
    R = 0.0
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        w, r = weights[i], positions[i, axis]
        R += w * r
    return R


def barycenter_component(indices, axis : int, weights = None):
    if axis < 0 or axis > 2:
        raise ValueError('Component out of range, provide an integer from 0 to 2.')
    #
    if weights is None:
        def reaction_coordinate(rs):
            return _barycenter_component(rs, axis)
    else:
        ws = weights / sum(weights)
        def reaction_coordinate(rs):
            return _weighted_barycenter_component(rs, axis, ws)
    #
    ξ, Jξ = value_and_jac_over(indices)(reaction_coordinate)
    return CollectiveVariableData(1, jit(ξ), jit(Jξ))
