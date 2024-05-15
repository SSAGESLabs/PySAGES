#!/usr/bin/env python
# the ermsd calculation code is taken from barnaba package
# github.com/srnas/barnaba

import numpy as np
from scipy.spatial import distance

f_factors = [0.5, 0.5, 0.3]
scale = [1.0 / f_factors[0], 1.0 / f_factors[1], 1.0 / f_factors[2]]


def calc_lcs(coords):
    """
    Calculate local coordinates system

    Calculate origin position and xyz orthonormal vectors
    in the six-member rings

    Parameters
    ----------
    coords : (3, n, 3) numpy array
        positions of C2,C4 and C6 atoms for pyrimidines (C,U,T)
        and C2,C6,C4 for purines (A,G) (axis 0)
        relative to n nucleobases (axis 1).
        xyz coordinates in axis 2.

    Returns
    -------
    lcs : (n,3,3) numpy array
        x y z vectors defining a local coordinate system
        in the geometrical center of the nucleobase
    origo : (n,3) numpy array
        origin  of local coordinate systems
    """

    # calculate center of mass
    origo = np.sum(coords, axis=0) / 3.0

    # CoM-C2 (x axis)
    x = coords[0] - origo
    x_norm = np.sqrt(np.sum(x * x, axis=1))
    x = x / x_norm[:, np.newaxis]
    # CoM-C4/C6
    c = coords[1] - origo
    # z/y axis
    z = np.cross(x, c, axis=1)
    z_norm = np.sqrt(np.sum(z * z, axis=1))
    z = z / z_norm[:, np.newaxis]

    y = np.cross(z, x, axis=1)
    lcs = np.array([x.T, y.T, z.T]).T
    return lcs, origo


def calc_3dmat(coords, cutoff):
    """
    Calculate relative position of nucleobases within an ellipsoidal cutoff

    Parameters
    ----------
    coords : (3, n,3) numpy array
        positions of C2,C4 and C6 atoms for pyrimidines (C,U,T)
        and C2,C6,C4 for purines (A,G) (axis 0)
        relative to n nucleobases (axis 1).
        xyz coordinates in axis 2.

    cutoff : float
       ellipsoidal cutoff

    Returns
    -------
    dotp : (x,3) numpy array
       xyz coordinates for each pair

    m_idx : (x,2) numpy array
       indeces of the pair
    """

    # prune search first
    lcs, origo = calc_lcs(coords)
    max_r = np.max(f_factors) * cutoff
    dmat = distance.squareform(distance.pdist(origo))
    m_idx = np.array(np.where((dmat < max_r) & (dmat > 0.01))).T

    # calculate scaled distances
    diff = origo[m_idx[:, 1]] - origo[m_idx[:, 0]]

    dotp = np.array([np.dot(diff[i], lcs[j]) for i, j in zip(range(len(diff)), m_idx[:, 0])])
    return dotp, m_idx


def calc_gmat(coords, cutoff):
    """
    Calculate G-vectors for each pair of bases
    within ellipsoidal cutoff distance

    Parameters
    ----------
    coords : (3,n,3) numpy array
        positions of C2,C4 and C6 atoms for pyrimidines (C,U,T)
        and C2,C6,C4 for purines (A,G) (axis 0)
        relative to n nucleobases (axis 1)
        xyz coordinates in axis 2.

    cutoff : float
        ellipsoidal cutoff

    Returns
    -------
    dotp : (n,n,4) numpy array
        G coordinates for each pair.
        For pairs outside the cutoff the coordinates are (0,0,0,0)
    """

    ll = coords.shape[1]

    mat = np.zeros((ll, ll, 4))

    dotp, m_idx = calc_3dmat(coords, cutoff)

    # return zero matrix when there are no contacts
    if dotp.shape[0] == 0:
        return mat
    dotp *= np.array(scale)[np.newaxis, :]
    dotp_norm = np.sqrt(np.sum(dotp**2, axis=1))

    # calculate 4D g-vector
    ff = (np.pi * dotp_norm) / cutoff
    factor13 = np.sin(ff) / ff
    factor4 = ((1.0 + np.cos(ff)) * cutoff) / np.pi
    gmat = dotp * factor13[:, np.newaxis]
    gmat = np.concatenate((gmat, factor4[:, np.newaxis]), axis=1)

    # set to zero when norm is larger than cutoff
    gmat[dotp_norm > cutoff] = 0.0
    mat[m_idx[:, 0], m_idx[:, 1]] = gmat

    return mat


def ermsd(reference, coords, cutoff):
    ref_mat = calc_gmat(reference, cutoff).reshape(-1)
    gmat = calc_gmat(coords, cutoff).reshape(-1)
    return np.sqrt(np.sum((ref_mat - gmat) ** 2) / coords.shape[1])


base_coords = np.loadtxt("base_coords.txt")  # in (n_traj, 3*n*3) shape
ermsd_barnaba = []
n_residues = int(base_coords.shape[1] / 9)
ref_coords = np.loadtxt("reference.txt").reshape(3, n_residues, 3, order="F")
# notice here the order 'F' is due to the weird ordering that barnaba is using
for coords in base_coords:
    coords_reshaped = coords.reshape(3 * n_residues, 3).reshape(3, n_residues, 3, order="F")
    # tricky reshape! first reshape to the same dimension as the reference.txt,
    # then do Fortran reshape.
    # basically we need to make sure the axis are correct:
    # base atom, nucleotides, xyz
    ermsd_barnaba.append(ermsd(ref_coords, coords_reshaped, cutoff=3.2))

ermsd_barnaba = np.array(ermsd_barnaba)
ermsd_pysages = np.loadtxt("ermsd.txt")

assert (
    np.mean((ermsd_pysages - ermsd_barnaba) ** 2) < 1e-4
), "the difference between pysages ermsd and barnaba version is too large!"

print("checking for rmsd passed!")
