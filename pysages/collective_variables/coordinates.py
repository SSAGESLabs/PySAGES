# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import jax.numpy as np
from jax.numpy import linalg
from .core import TwoPointCV, AxisCV


def barycenter(positions):
    """
    Returns the geometric center, or centroid, of a group of points in space.

    Parameters
    ----------
    positions : DeviceArray
        Array containing the positions of the points for which to compute the barycenter.

    Returns
    -------
    barycenter : DeviceArray
        3D array with the barycenter coordinates.

    """
    return np.sum(positions, axis=0) / positions.shape[0]


def weighted_barycenter(positions, weights):
    """
    Returns the center of a group of points in space weighted by arbitrary weights.

    Parameters
    ----------
    positions : DeviceArray
        Array containing the positions of the points for which to compute the barycenter.
    weights : DeviceArray
        Array containing the weights to be used when computing the barycenter.

    Returns
    -------
    weighted_barycenter : DeviceArray
        3D array with the weighted barycenter coordinates.

    """
    n = positions.shape[0]
    R = np.zeros(3)
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        w, r = weights[i], positions[i]
        R += w * r
    return R


class Component(AxisCV):
    def __init__(self, indices, axis):
        super().__init__(indices, axis)
        self.requires_box_unwrapping = True

    @property
    def function(self):
        return (lambda rs: barycenter(rs)[self.axis])


class Distance(TwoPointCV):
    @property
    def function(self):
        return distance


def distance(r1, r2):
    """
    Returns the distance between two points in space.

    Parameters
    ----------
    r1 : DeviceArray
        Array containing the position in space of point 1.
    r2 : DeviceArray
        Array containing the position in space of point 2.

    Returns
    -------
    distance : float
        Distance between the two points.

    """
    return linalg.norm(r1 - r2)
