#!/usr/bin/env python3
# Collective variable funnel
import jax.numpy as np
from jax.numpy import linalg

from pysages.colvars.coordinates import barycenter, weighted_barycenter
from pysages.colvars.core import CollectiveVariable


def fitted_positions(positions, references, weights):
    if weights is None:
        pos_b = barycenter(positions)
        ref_b = barycenter(references)
        fit_pos = np.add(positions, -pos_b)
        fit_ref = np.add(references, -ref_b)
    else:
        pos_b = weighted_barycenter(positions, weights)
        ref_b = weighted_barycenter(references, weights)
        fit_pos = positions * weights - pos_b
        fit_ref = references * weights - ref_b
    return fit_pos, fit_ref


def kabsch(P0, Q0, weights):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """
    P, Q = fitted_positions(P0, Q0, weights)
    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = linalg.svd(C, full_matrices=False)
    dh = linalg.det(V) * linalg.det(W)
    S = S.at[-1].multiply(np.sign(dh))
    V = V.at[:, -1].multiply(np.sign(dh))
    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


class Projection_on_Axis_mobile(CollectiveVariable):
    """
    Use a reference to calculate the projection on an axis of a set of atoms.
    The algorithm is based on https://doi.org/10.1002/jcc.20110.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.

    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The coordinates must match the ones of the atoms used in indices.
    weights_lig: list[tuple(float)]
       weights of the mobile group of particles
    weights_prot: list[tuple(float)]
       weights of the reference group of particles
    A: list[tuple(float)]
       Initial vector of the axis
    B: list[tuple(float)]
       Final vector of the axis
    box: list[tuple(float)]
       Box vector of the axis (only works for orthorhombic boxes for now)
    """

    def __init__(
        self,
        indices,
        references,
        weights_lig=None,
        weights_prot=None,
        A=[0, 0, 0],
        B=[0, 0, 1],
        box=[1.0, 1.0, 1.0],
    ):
        super().__init__(indices, 3)
        self.requires_box_unwrapping = True
        self.references = np.asarray(references)
        self.weights_lig = np.asarray(weights_lig) if weights_lig else None
        self.weights_prot = np.asarray(weights_prot) if weights_prot else None
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.box = np.asarray(box)

    @property
    def function(self):
        return lambda r1, r2, r3: projection_mobile(
            r1,
            r2,
            r3,
            self.references,
            self.weights_lig,
            self.weights_prot,
            self.A,
            self.B,
            self.box,
        )


def center(positions, weights):
    if weights is None:
        return barycenter(positions)
    else:
        return weighted_barycenter(positions, weights)


def periodic(distance, box):
    return np.mod(distance + box * 0.5, box) - 0.5 * box


def projection_mobile(
    ligand, backbone, pos_anchor, references, weights_lig, weights_prot, A, B, box
):
    ligand_distances = periodic(ligand - pos_anchor, box)
    new_lig_pos = pos_anchor + ligand_distances
    com_lig = center(new_lig_pos, weights_lig)
    com_prot = center(backbone, weights_prot)
    com_ref = center(references, weights_prot)
    lig_rot = np.dot(com_lig - com_prot, kabsch(backbone, references, weights_prot)) + com_ref
    vector = B - A
    norm = linalg.norm(vector)
    eje = vector / norm
    return np.dot(eje, lig_rot - A)


class Perp_projection_on_Axis_mobile(CollectiveVariable):
    """
    Use a reference to calculate the perpendicular projection on an axis of a set of atoms.
    The algorithm is based on https://doi.org/10.1002/jcc.20110.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.

    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The coordinates must match the ones of the atoms used in indices.
    weights_lig: list[tuple(float)]
       weights of the mobile group of particles
    weights_prot: list[tuple(float)]
       weights of the reference group of particles
    A: list[tuple(float)]
       Initial vector of the axis
    B: list[tuple(float)]
       Final vector of the axis
    box: list[tuple(float)]
       Box vector of the axis (only works for orthorhombic boxes for now)
    """

    def __init__(
        self,
        indices,
        references,
        weights_lig=None,
        weights_prot=None,
        A=[0, 0, 0],
        B=[0, 0, 1],
        box=[1.0, 1.0, 1.0],
    ):
        super().__init__(indices, 3)
        self.requires_box_unwrapping = True
        self.references = np.asarray(references)
        self.weights_lig = np.asarray(weights_lig) if weights_lig else None
        self.weights_prot = np.asarray(weights_prot) if weights_prot else None
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.box = np.asarray(box)

    @property
    def function(self):
        return lambda r1, r2, r3: perp_projection_mobile(
            r1,
            r2,
            r3,
            self.references,
            self.weights_lig,
            self.weights_prot,
            self.A,
            self.B,
            self.box,
        )


def perp_projection_mobile(
    ligand, backbone, pos_anchor, references, weights_lig, weights_prot, A, B, box
):
    ligand_distances = periodic(ligand - pos_anchor, box)
    new_lig_pos = pos_anchor + ligand_distances
    com_lig = center(new_lig_pos, weights_lig)
    com_prot = center(backbone, weights_prot)
    com_ref = center(references, weights_prot)
    lig_rot = np.dot(com_lig - com_prot, kabsch(backbone, references, weights_prot)) + com_ref
    vector = B - A
    norm = linalg.norm(vector)
    eje = vector / norm
    perp = (lig_rot - A) - np.dot(eje, lig_rot - A) * eje
    return linalg.norm(perp)
