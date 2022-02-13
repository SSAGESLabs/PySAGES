# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES
"""
Collective variable angles describe the angle spanning by 3 (or 4 for dihedral) particles in the simulation.

It is common to describe such anlges inside a molecule or protein characteristic for a conformation change.
"""

import jax.numpy as np
from jax.numpy import linalg
from .core import ThreePointCV, FourPointCV


class Angle(ThreePointCV):
    """Angle between 3 points in space.

    Angle collective variables calcualte the angle spanned by three points in space (usualy atom positions).
    Take a look at the `pysages.collective_variables.core.ThreePointCV` for details on the constructor.
    """
    @property
    def function(self):
        """
        Function generator

        Returns
        -------
        Function that calculate the angle value from a simlation snap shot.
        Look at `pysages.collective_variables.angles.angle` for details.
        """
        return angle


def angle(pos1, pos2, pos3):
    """Function to calculate angle between 3 points.

    Takes 3 positions in space and calculates the angle between them.

    :math:`\\vec{q} = \\vec{p}_1 - \\vec{p}_2`

    :math:`\\vec{r} = \\vec{p}_3 - \\vec{p}_2`

    :math:`\\theta = \\arctan_2(|\\vec{q} \\times \\vec{r}|, \\vec{q} \\cdot \\vec{r})`

    Parameters
    ----------
    pos1: DeviceArray
       :math:`\\vec{p}_1` 3D vector in space
    pos1: DeviceArray
       :math:`\\vec{p}_2` 3D vector in space
    pos1: DeviceArray
       :math:`\\vec{p}_3` 3D vector in space

    Returns
    -------
    float
       :math:`\\theta`
    """
    qvec = pos1 - pos2
    rvec = pos3 - pos2
    return np.arctan2(linalg.norm(np.cross(qvec, rvec)), np.dot(qvec, rvec))


class DihedralAngle(FourPointCV):
    """
    Angle collective variables calcualte the dihedral angle spanned by four points in space (usualy atom positions).
    Take a look at the `pysages.collective_variables.core.FourPointCV` for details on the constructor.
    """
    @property
    def function(self):
        """
        Returns
        -------
        Returns the function that calculate the dihedral angle value from a simlation snap shot.
        Look at `pysages.collective_variables.angles.dihedral_angle` for details.
        """
        return dihedral_angle


def dihedral_angle(pos1, pos2, pos3, pos4):
    """
    Calculate dihedral angle between 4 points in space.

    Takes 4 positions in space and calculates the dihedral angle.

    :math:`\\vec{q} = \\vec{p}_3 - \\vec{p}_2`

    :math:`\\vec{r} = (\\vec{p}_2 - \\vec{p}_1) \\times \\vec{q}`

    :math:`\\vec{s} =  \\vec{q} \\times ( \\vec{p}_4  - \\vec{p}_3 )`

    :math:`\\theta = \\arctan_2( (\\vec{r} \\times \\vec{s}) \\cdot \\vec{q}, |\\vec{q}| \\vec{r} \\cdot \\vec{s})`

    Parameters
    ----------
    pos1: DeviceArray
       :math:`\\vec{p}_1` 3D vector in space
    pos2: DeviceArray
       :math:`\\vec{p}_2` 3D vector in space
    pos3: DeviceArray
       :math:`\\vec{p}_3` 3D vector in space
    pos4: DeviceArray
       :math:`\\vec{p}_4` 3D vector in space
    Returns
    -------
    float
       :math:`\\theta`
    """
    qvec = pos3 - pos2
    rvec = np.cross(pos2 - pos1, qvec)
    svec = np.cross(qvec, pos4 - pos3)
    return np.arctan2(np.dot(np.cross(rvec, svec), qvec), np.dot(rvec, svec) * linalg.norm(qvec))
