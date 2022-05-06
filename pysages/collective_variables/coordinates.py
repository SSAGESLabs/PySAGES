# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective Variables that are computed from the Cartesian coordinates.
"""

from jax import numpy as np
from jax.numpy import linalg

from pysages.collective_variables.core import TwoPointCV, AxisCV


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
    group_length = positions.shape[0]
    center = np.zeros(3)
    # TODO: Replace by `np.sum` and `vmap`  # pylint:disable=fixme
    for i in range(group_length):
        w, p = weights[i], positions[i]
        center += w * p
    return center


class Component(AxisCV):
    """
    Use a specific cartesian component of the center of mass of the group of atom selected
    via the indices.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices. From each group the barycenter is calculated.
    axis: int
       Cartesian coordinate axis component `0` (X), `1` (Y), `2` (Z) that is requested as CV.
    """

    @property
    def function(self):
        return lambda rs: barycenter(rs)[self.axis]


class Distance(TwoPointCV):
    """
    Use the distance of atom groups selected via the indices as collective variable.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices. (2 Groups required)
    """

    @property
    def function(self):
        return distance


def distance(p1, p2):
    """
    Returns the distance between two points in space.

    Parameters
    ----------
    p1 : DeviceArray
        Array containing the position in space of the first point.
    p2 : DeviceArray
        Array containing the position in space of the second point.

    Returns
    -------
    distance : float
        Distance between the two points.
    """
    return linalg.norm(p1 - p2)
