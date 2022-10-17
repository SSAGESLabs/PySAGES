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

from pysages.colvars.coordinates import barycenter
from pysages.colvars.core import CollectiveVariable, FourPointCV, ThreePointCV


class Angle(ThreePointCV):
    """Angle between 3 points in space.

    Angle collective variables calculated as the angle spanned by three points in space
    (usually atom positions).
    Take a look at the `pysages.colvars.core.ThreePointCV` for details on the
    constructor.
    """

    @property
    def function(self):
        """
        Function generator

        Returns
        -------
        Function that calculates the angle value from a simulation snapshot.
        Look at `pysages.colvars.angles.angle` for details.
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
    Take a look at the `pysages.colvars.core.FourPointCV` for details on the
    constructor.
    """

    @property
    def function(self):
        """
        Returns
        -------
        Function that calculates the dihedral angle value from a simulation snapshot.
        Look at `pysages.colvars.angles.dihedral_angle` for details.
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


class PhaseAngle(CollectiveVariable):
    """
    Computes the phase angle of a monocyclic ring by the Cremer-Pople method.
    Mathematical definitions can be found in
    [D. Cremer and J. A. Pople, JACS, 1974](https://pubs.acs.org/doi/10.1021/ja00839a011)
    Equations 4-14.
    Notice that for rings with more than 5 atoms, there are `(N - 1) / 2 - 1`
    phase angles, and this class only calculates the first one (`m = 2` in
    Equations 12 and 13, or see `pysages.colvars.angles.phase_angle` for math).
    Also, the phase angle obtained via the Cremer-Pople method can be converted
    to the Altona-Sundaralingam order parameter by adding `pi / 2` to the result
    and then converting to degrees.

    Usage
    -------
    cvs = [PhaseAngle([index1, index2, ...])]

    Notice that the phase angle is dependent on the order of the indices. For example,
    the convention for sugar pucker of ribose in RNA/DNA is: O4', C1', C2', C3', C4'.
    """

    @property
    def function(self):
        """
        Returns
        -------
        Function that calculates the dihedral angle value from a simulation snapshot.
        Look at `pysages.colvars.angles.phase_angle` for details.
        """
        return phase_angle


def phase_angle(rs):
    r"""
    calculate phase angle (first phase angle if N>5) based on Cremer-Pople method.
    :math:`r0 = 1/N \sum_i^N \vec{r}_i`
    :math:`\vec{R1} = \sum_i^N (\vec{r}_i -r_c) \sin(2\pi (i-1)/N)`
    :math:`\vec{R2} = \sum_i^N (\vec{r}_i -r_c) \cos(2\pi (i-1)/N)`
    :math:`\hat{n} = \vec{R1} \times \vec{R2}/ (|\vec{R1}\times\vec{R2}|)`
    :math:`z_i = (\vec_{r}_i-r_c) \cdot \hat{n}`
    :math:`a =  \sqrt(2/N) \sum_i^N z_i \cos(2\pi 2(i-1)/N)`
    :math:`b = -\sqrt(2/N) \sum_i^N z_i \sin(2\pi 2(i-1)/N)`
    :math:`P=\atan2(b/a)`

    Parameters
    ------------
    rs: DeviceArray
        :math: `\vec{r}_i` array of 3D vector in space

    Returns
    ------------
    float
        :math:`P` in range -pi to pi.
    """
    N = len(rs)
    r0 = barycenter(rs)
    rc = rs - r0
    theta = 2j * np.pi * np.arange(N) / N
    fourier_coeff = np.exp(theta)
    R1 = np.dot(rc.T, np.imag(fourier_coeff))
    # Notice the imag part corresponds to sin. The order of R1/R2 matters
    # because otherwise the n would be inverted.
    R2 = np.dot(rc.T, np.real(fourier_coeff))
    n = np.cross(R1, R2)
    n /= linalg.norm(n)
    z = np.dot(rc, n)
    fourier_coeff2 = np.exp(2 * theta)
    a = np.sqrt(2 / N) * np.sum(z * np.real(fourier_coeff2))
    b = -np.sqrt(2 / N) * np.sum(z * np.imag(fourier_coeff2))
    return np.arctan2(b, a)
