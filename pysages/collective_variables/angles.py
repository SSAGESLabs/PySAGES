# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective variable angles describe the angle spanning by 3 (or 4 for dihedral) particles
in the simulation.

It is common to describe such angles inside a molecule or protein characteristic for a
conformation change.
"""

from jax import numpy as np
from jax.numpy import linalg
from pysages.collective_variables.core import ThreePointCV, FourPointCV


class Angle(ThreePointCV):
    """Angle between 3 points in space.

    Angle collective variables calculated as the angle spanned by three points in space
    (usually atom positions).
    Take a look at the `pysages.collective_variables.core.ThreePointCV` for details on the
    constructor.
    """

    @property
    def function(self):
        """
        Function generator

        Returns
        -------
        Function that calculates the angle value from a simulation snapshot.
        Look at `pysages.collective_variables.angles.angle` for details.
        """
        return angle


def angle(p1, p2, p3):
    r"""
    Calculates angle between 3 points in space.

    Takes 3 positions in space and calculates the angle between them.

    :math:`\vec{q} = \vec{p}_1 - \vec{p}_2`

    :math:`\vec{r} = \vec{p}_3 - \vec{p}_2`

    :math:`\theta = \atan2(|\vec{q} \times \vec{r}|, \vec{q} \cdot \vec{r})`

    Parameters
    ----------
    p1: DeviceArray
       :math:`\vec{p}_1` 3D vector in space
    p2: DeviceArray
       :math:`\vec{p}_2` 3D vector in space
    p3: DeviceArray
       :math:`\vec{p}_3` 3D vector in space

    Returns
    -------
    float
       :math:`\theta`
    """
    q = p1 - p2
    r = p3 - p2
    return np.arctan2(linalg.norm(np.cross(q, r)), np.dot(q, r))


class DihedralAngle(FourPointCV):
    """
    Computes the dihedral angle spanned by four points in space (usually atom positions).
    Take a look at the `pysages.collective_variables.core.FourPointCV` for details on the
    constructor.
    """

    @property
    def function(self):
        """
        Returns
        -------
        Function that calculates the dihedral angle value from a simulation snapshot.
        Look at `pysages.collective_variables.angles.dihedral_angle` for details.
        """
        return dihedral_angle


def dihedral_angle(p1, p2, p3, p4):
    r"""
    Calculate dihedral angle between 4 points in space.

    Takes 4 positions in space and calculates the dihedral angle.

    :math:`\vec{q} = \vec{p}_3 - \vec{p}_2`

    :math:`\vec{r} = (\vec{p}_2 - \vec{p}_1) \times \vec{q}`

    :math:`\vec{s} = \vec{q} \times (\vec{p}_4  - \vec{p}_3)`

    :math:`\theta = \atan2((\vec{r} \times \vec{s}) \cdot \vec{q}, |\vec{q}| \vec{r} \cdot \vec{s})`

    Parameters
    ----------
    p1: DeviceArray
       :math:`\vec{p}_1` 3D vector in space
    p2: DeviceArray
       :math:`\vec{p}_2` 3D vector in space
    p3: DeviceArray
       :math:`\vec{p}_3` 3D vector in space
    p4: DeviceArray
       :math:`\vec{p}_4` 3D vector in space

    Returns
    -------
    float
       :math:`\theta`
    """
    q = p3 - p2
    r = np.cross(p2 - p1, q)
    s = np.cross(q, p4 - p3)
    return np.arctan2(np.dot(np.cross(r, s), q), np.dot(r, s) * linalg.norm(q))
