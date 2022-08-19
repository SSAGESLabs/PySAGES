# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective variables for orientiations describe the orientation measures
of particles in the simulation respect to a reference.

It is common to describe such orientations using RMSD, tilt, sping angle
and more to spot inside a molecule or protein a conformation change.
"""

from jax import numpy as np
from jax.numpy import linalg

from pysages.colvars.core import CollectiveVariable, AxisCV
from pysages.colvars.coordinates import barycenter, weighted_barycenter


_QUATERNION_BASES = {
    0: (0, 1, 0, 0),
    1: (0, 0, 1, 0),
    2: (0, 0, 0, 1),
}


def quaternion_matrix(positions, references, weights):
    """
    Function to construct the quaternion matrix based on the reference positions.

    Parameters
    ----------
    positions: np.array
       atomic positions via indices.
    references: np.array
       Cartesian coordinates of the reference positions of the atoms in indices.
       The number of coordinates must match the ones of the atoms used in indices.
    weights: np.array
       Weights for the barycenter calculation
    """
    if len(positions) != len(references):
        raise RuntimeError("References must be of the same length as the positions")
    pos_b = np.where(weights is None,
                     barycenter(positions),
                     weighted_barycenter(positions, weights))
    ref_b = np.where(weights is None,
                     barycenter(references),
                     weighted_barycenter(references, weights))
    R = np.zeros((3, 3))
    if weights is None:
        for pos, ref in zip(positions, references):
            R += np.outer(pos - pos_b, ref - ref_b)
    else:
        for pos, ref, w in zip(positions, references, weights):
            R += np.outer(w*pos - pos_b, w*ref - ref_b)
    S_00 = R[0, 0] + R[1, 1] + R[2, 2]
    S_01 = R[1, 2] - R[2, 1]
    S_02 = R[2, 0] - R[0, 2]
    S_03 = R[0, 1] - R[1, 0]
    S_11 = R[0, 0] - R[1, 1] - R[2, 2]
    S_12 = R[0, 1] + R[1, 0]
    S_13 = R[0, 2] + R[2, 0]
    S_22 = -R[0, 0] + R[1, 1] - R[2, 2]
    S_23 = R[1, 2] + R[2, 1]
    S_33 = -R[0, 0] + R[1, 1] + R[2, 2]
    S = np.array(
        [
            [S_00, S_01, S_02, S_03],
            [S_01, S_11, S_12, S_13],
            [S_02, S_12, S_22, S_23],
            [S_03, S_13, S_23, S_33],
        ]
    )
    return S


class Tilt(AxisCV):
    """
    Use a reference to fit the tilt rotation respect to an axis.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.
    axis: int
       Cartesian coordinate axis of rotation `0` (X), `1` (Y), `2` (Z) that is requested in CV.
    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices. The number
       of coordinates must match the ones of the atoms used in indices.
    """

    def __init__(self, indices, axis, references, weights=None):
        if weights is not None and len(indices) != len(weights):
            raise RuntimeError("Indices and weights must be of the same length")
        super().__init__(indices, axis)
        self.references = np.asarray(references)
        self.weights = np.asarray(weights)
        self.e = np.asarray(_QUATERNION_BASES[axis])

    @property
    def function(self):
        return lambda rs: tilt(rs, self.references, self.axis, self.weights)


def tilt(r1, r2, e, w):
    """
    Function to calculate the tilt rotation respect to an axis.

    Parameters
    ----------
    r1: JaxArray
       Atomic positions.
    r2: JaxArray
       Cartesian coordinates of the reference position of the atoms in indices.
       The number of coordinates must match the ones of the atoms used in indices.
    e: JaxArray
       Quaternion rotation axis.
    w: JaxArray
       Atomic weights.
    """
    S = quaternion_matrix(r1, r2, w)
    _, v = linalg.eigh(S)
    v_dot_e = np.dot(v[:, 3], e)
    cs = np.cos(np.arctan2(v_dot_e, v[0, 3]))
    z = np.where(cs == 0, 0, v[0, 3] / cs)
    return 2 * z * z - 1


class RotationEigenvalues(CollectiveVariable):
    """
    Calculate the eigenvalues of the rotation matrix respect to a reference.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.
    axis: int
       Index for the eigenvalue of the rotation matrix (a value form `0` to `3`).
    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The number of coordinates must match the ones of the atoms used in indices.
    """

    def __init__(self, indices, axis, references, weights=None):
        if weights is not None and len(indices) != len(weights):
            raise RuntimeError("Indices and weights must be of the same length")
        super().__init__(indices)
        self.axis = axis
        self.references = np.asarray(references)
        self.weights = np.asarray(weights)

    @property
    def function(self):
        return lambda r: rotation_eigvals(r, self.references, self.weights)[self.axis]


def rotation_eigvals(r1, r2, w):
    """
    Returns the eigenvalue of the quaternion matrix rspect to an axis.

    Parameters
    ----------
    r1: np.array
       Atomic positions.
    axis: int
       select the eigenvalue of the matrix.
    r2: np.array
       Cartesian coordinates of the reference position of the atoms in r1.
    """
    S = quaternion_matrix(r1, r2, w)
    return linalg.eigvalsh(S)


class SpinAngle(AxisCV):
    """
    Use a reference to fit the spin angle rotation respect to an axis.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.
    axis: int
       Cartesian coordinate axis of rotation `0` (X), `1` (Y), `2` (Z) that is requested in CV.
    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The coordinates must match the ones of the atoms used in indices.
    """

    def __init__(self, indices, axis, references, weights=None):
        if weights is not None and len(indices) != len(weights):
            raise RuntimeError("Indices and weights must be of the same length")
        super().__init__(indices, axis)
        self.references = np.asarray(references)
        self.e = np.array(_QUATERNION_BASES[axis])
        self.weights = np.asarray(weights)

    @property
    def function(self):
        return lambda r: spin(r, self.references, self.e, self.weights)


def spin(r1, r2, e, w):
    """
    Calculate the spin angle rotation respect to an axis.

    Parameters
    ----------
    r1: np.array
       Atomic positions.
    r2: np.array
       Cartesian coordinates of the reference position of the atoms.
    e: np.array
        Rotation axis.
    """
    S = quaternion_matrix(r1, r2, w)
    _, v = linalg.eigh(S)
    v_dot_e = np.dot(v[:, 3], e)
    return 2 * np.arctan2(v_dot_e, v[0, 3])


class RotationAngle(CollectiveVariable):
    """
    Use a reference to fit the rotation angle of the atoms.
    The angle varies from -pi to pi.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.
    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The coordinates must match the ones of the atoms used in indices.
    """

    def __init__(self, indices, references, weights=None):
        if weights is not None and len(indices) != len(weights):
            raise RuntimeError("Indices and weights must be of the same length")
        super().__init__(indices)
        self.references = np.asarray(references)
        self.weights = np.asarray(weights)

    @property
    def function(self):
        return lambda r: rotation_angle(r, self.references, self.weights)


def rotation_angle(r1, r2, w):
    """
    Calculate the rotation angle respect to a reference.

    Parameters
    ----------
    r1: np.array
       Atomic positions.
    r2: np.array
       Cartesian coordinates of the reference position of the atoms.
    """
    S = quaternion_matrix(r1, r2, w)
    _, v = linalg.eigh(S)
    return 2 * np.arccos(v[0, 3])


class RotationProjection(CollectiveVariable):
    """
    Use a reference to fit the cosine of the angle rotation of the atoms.
    The projection varies from -1 to 1.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.
    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The coordinates must match the ones of the atoms used in indices.
    """

    def __init__(self, indices, references, weights=None):
        if weights is not None and len(indices) != len(weights):
            raise RuntimeError("Indices and weights must be of the same length")
        super().__init__(indices)
        self.references = np.asarray(references)
        self.weights = np.asarray(weights)

    @property
    def function(self):
        return lambda r: rotation_projection(r, self.references, self.weights)


def rotation_projection(r1, r2, w):
    """
    Calculate the rotation angle projection.

    Parameters
    ----------
    r1: np.array
       Atomic positions.
    r2: np.array
       Cartesian coordinates of the reference position of the atoms.
    """
    S = quaternion_matrix(r1, r2, w)
    _, v = linalg.eigh(S)
    return 2 * v[0, 3] * v[0, 3] - 1


class RMSD(CollectiveVariable):
    """
    Use a reference to calculate the RMSD of a set of atoms.
    The algorithm is based on https://doi.org/10.1002/jcc.20110.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.
    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The coordinates must match the ones of the atoms used in indices.
    """

    def __init__(self, indices, references, weights=None):
        if weights is not None and len(indices) != len(weights):
            raise RuntimeError("Indices and weights must be of the same length")
        super().__init__(indices)
        self.references = np.asarray(references)
        self.weights = np.asarray(weights)

    @property
    def function(self):
        return lambda r: rmsd(r, self.references, self.weights)


def sq_norm_rotation(positions, references, weights):
    """
    Calculate the squared norm of the atomic positions and references respect to barycenters.

    Parameters
    ----------
    positions: np.array
       Atomic positions.
    references: np.array
       Cartesian coordinates of the reference position of the atoms.
    """
    if len(positions) != len(references):
        raise RuntimeError("References must be of the same length as the positions")
    pos_b = np.where(weights is None,
                     barycenter(positions),
                     weighted_barycenter(positions, weights))
    ref_b = np.where(weights is None,
                     barycenter(references),
                     weighted_barycenter(references, weights))
    R = 0.0
    if weights is None:
        for pos, ref in zip(positions, references):
            pos -= pos_b
            ref -= ref_b
            R += np.dot(pos, pos) + np.dot(ref, ref)
    else:
        for pos, ref, w in zip(positions, references, weights):
            pos *= w
            ref *= w
            pos -= pos_b
            ref -= ref_b
            R += np.dot(pos, pos) + np.dot(ref, ref)
    return R


def rmsd(r1, r2, w_0):
    """
    Calculate the rmsd respect to a reference using quaternions.

    Parameters
    ----------
    r1: np.array
       Atomic positions.
    r2: np.array
       Cartesian coordinates of the reference position of the atoms.
    """
    N = r1.shape[0]
    S = quaternion_matrix(r1, r2, w_0)
    w = linalg.eigvalsh(S)
    norm_sq = sq_norm_rotation(r1, r2, w_0)
    return np.sqrt((norm_sq - 2 * np.max(w)) / N)


class QuaternionComponent(CollectiveVariable):
    """
    Calculate the quaternion component of the rotation respect to a reference.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.
    axis: int
       Index for the component of the quaternion (a value form `0` to `3`).
    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The number of coordinates must match the ones of the atoms used in indices.
    """

    def __init__(self, indices, axis, references, weights=None):
        if weights is not None and len(indices) != len(weights):
            raise RuntimeError("Indices and weights must be of the same length")
        super().__init__(indices)
        self.axis = axis
        self.references = np.asarray(references)
        self.weights = np.asarray(weights)

    @property
    def function(self):
        return lambda r: quaternion_component(r, self.references, self.weights)[self.axis]


def quaternion_component(r1, r2, w):
    """
    Returns the eigenvalue of the quaternion matrix rspect to an axis.

    Parameters
    ----------
    r1: np.array
       Atomic positions.
    axis: int
       select the component of the quaternion for rotation.
    r2: np.array
       Cartesian coordinates of the reference position of the atoms in r1.
    """
    S = quaternion_matrix(r1, r2, w)
    _, v = linalg.eigh(S)
    return v
