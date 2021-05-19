# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)
# Copyright (c) 2021: Ludwig Schneider (see LICENSE.md)

import jax.numpy as np
from jax.numpy import linalg
from .core import CollectiveVariable, AxisCV


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


class ShapeAnisotropy(CollectiveVariable):
    @property
    def function(self):
        return shape_anisotropy


def shape_anisotropy(positions):
    λ1, λ2, λ3 = principal_moments(positions)
    return (3 * (λ1**2 + λ2**2 + λ3**2) / (λ1 + λ2 + λ3)**2 - 1) / 2
