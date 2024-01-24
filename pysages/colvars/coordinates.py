# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective Variables that are computed from the Cartesian coordinates.
"""

from jax import numpy as np
from jax.numpy import linalg

from pysages.colvars.core import AxisCV, TwoPointCV, multicomponent


def barycenter(positions):
    """
    Returns the geometric center, or centroid, of a group of points in space.

    Parameters
    ----------
    positions : jax.Array
        Array containing the positions of the points for which to compute the barycenter.

    Returns
    -------
    barycenter : jax.Array
        3D array with the barycenter coordinates.
    """
    return np.sum(positions, axis=0) / positions.shape[0]


def weighted_barycenter(positions, weights):
    """
    Returns the center of a group of points in space weighted by arbitrary weights.

    Parameters
    ----------
    positions : jax.Array
        Array containing the positions of the points for which to compute the barycenter.
    weights : jax.Array
        Array containing the weights to be used when computing the barycenter.

    Returns
    -------
    weighted_barycenter : jax.Array
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
    Use a specific Cartesian component of the center of mass of the group of atom selected
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
        if len(self.groups) == 0:
            return distance
        return lambda r1, r2: distance(barycenter(r1), barycenter(r2))


def distance(r1, r2):
    """
    Returns the distance between two points in space or
    between the barycenters of two groups of points in space.

    Parameters
    ----------
    r1: jax.Array
        Array containing the position in space of the first point or group of points.
    r2: jax.Array
        Array containing the position in space of the second point or group of points.

    Returns
    -------
    distance: float
        Distance between the two points.
    """

    return linalg.norm(r1 - r2)


@multicomponent
class Displacement(TwoPointCV):
    """
    Relative displacement between two points in space.

    Parameters
    ----------
    indices: Union[list[int], list[tuple(int)]]
        Indices of the reference atoms (two groups are required).
    """

    @property
    def function(self):
        if len(self.groups) == 0:
            return displacement
        return lambda r1, r2: displacement(barycenter(r1), barycenter(r2))


def displacement(r1, r2):
    """
    Displacement between two points in space or
    between the barycenters of two groups of points in space.

    Parameters
    ----------
    r1: jax.Array
        Array containing the position in space of the first point or group of points.
    r2: jax.Array
        Array containing the position in space of the second point or group of points.

    Returns
    -------
    displacement: jax.Array
        Displacement between the two points.
    """

    return r2 - r1
