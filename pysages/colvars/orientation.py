# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective variable for orientations describe the orientation measures
of particles in the simulations with respect to a reference

eRMSD describes distances between three-dimensional
RNA structures. The standard RMSD is not accurate enough for distinguish
RNA 3D structures. eRMSD provides a way to measuring the difference in
the base interacting network.
eRMSD also only requires the knowledge of C2, C4 and C6 for each base.

"""

from jax import numpy as np
from jax.experimental import host_callback
from jax.numpy import linalg

from pysages.colvars.coordinates import barycenter
from pysages.colvars.core import CollectiveVariable, multicomponent


@multicomponent
class ERMSD(CollectiveVariable):
    """
    Use a reference to calculate the eRMSD of a set of RNA structures.
    Mathematical details can be found in
    [S. Bottaro, F. Di Palma,
    and G. Bussi, NAR, 2014](https://doi.org/10.1093/nar/gku972)

    To use this method, the model has to have access to the coordinates of
    C2, C4, and C6 of each base.
    This is usually the case for all-atom simulations
    or coarse-grained models with SPQR-type of mappings.

    Parameters
    ------------
    indices: list[int].
        Must be a list of indices with dimension (n_residues*3,).
        The order is tricky,
        for Purines (A, G) the order in the tuple should be C2, C6, C4;
        for Pyrimidines (U, C) the order in the tuple should be C2, C4, C6.
        This weird order is simply due to the numbering of the base.
        Please make sure the order is correct,
        otherwise the results will be wrong.
    references: list[tuple(float)]
        Must be a list of tuple of floats, with dimension (n_residues*3, 3)
        The last dimension is the x,y,z of the
        cartesian coordinates of the reference position of the C2/C4/C6 atoms,
        with the same order as the indices.
    cutoff: float
        unitless cutoff. Default is 2.4
    a: float
        re-scaling scale in x and y direction. Default is 0.5 (nm).
    b: float
        re-scaling scale in z direction. Default is 0.3 (nm).
    """

    def __init__(self, indices, references, cutoff=2.4, a=0.5, b=0.3):
        super().__init__(indices)
        self.references = np.asarray(references)
        self.cutoff = cutoff
        self.a = a
        self.b = b

    @property
    def function(self):
        return lambda r: ermsd(r, self.references, self.cutoff, self.a, self.b)


def local_coordinates(rs):
    """
    calculate the local coordinate system for eRMSD calculation:
    origin and orthogonal x/y/z axis in the six membered rings.
    The x-axis is given by center of geometry (cog) -> C2,
    the z-axis is given by cross product
    between x-axis and cog -> second atom (C2 for U/C, C4 for A/G),
    and y-axis is given by cross product of z-axis and x-axis.

    Parameters
    ----------
    rs: (n_residues*3, 3) array
        positions of C2/4/6 atoms of every bases. The last dimension is x/y/z.
        The order is C2, C6, C4 for A/G and C2, C4, C6 for U/C.

    Returns
    ----------
    lcs: (n_residues, 3, 3) array
        x, y, z unit vectors of each base

    origins: (n_residues, 3) array
        origins (center of geometry) of the six membered rings of each base

    """
    n_sites, n_xyz = rs.shape
    assert n_xyz == 3, f"the number of coordinate dimension is {n_xyz}" " but should be 3!"
    assert n_sites % 3 == 0, (
        f"Please make sure the indices are in dimension " f" 3 * n_residues! Now it's {n_sites}."
    )
    n_residues = int(n_sites / 3)
    rs = rs.reshape(n_residues, 3, n_xyz)
    origins = np.average(rs, axis=1)
    # Cog->C2 (x axis)
    x = rs[:, 0] - origins
    x_hat = np.einsum("ij,i->ij", x, 1 / np.linalg.norm(x, axis=-1))
    # Cog->C4/C6
    cog_2nd_atom = rs[:, 1] - origins
    # z, y axis
    z = np.cross(x_hat, cog_2nd_atom, axis=1)
    z_hat = np.einsum("ij,i->ij", z, 1 / np.linalg.norm(z, axis=-1))

    y_hat = np.cross(z_hat, x_hat, axis=1)
    lcs = np.stack((x_hat, y_hat, z_hat), axis=1)

    return lcs, origins


def g_vector(lcs, origins, cutoff, a, b):
    r"""
    Calculate the smoothing function :math:`G(\tilde{r})`
    based on local coordinates and origins for eRMSD calculations

    Mathematical details can be found in
    [S. Bottaro, F. Di Palma,
    and G. Bussi, NAR, 2014](https://doi.org/10.1093/nar/gku972)

    :math:`\tilde{\mathbf{r}}=(\frac{r_x}{a}, \frac{r_y}{a}, \frac{r_z}{a})`

    :math:`\gamma = \pi/\tilde{r}_\mathrm{cutoff}`

    :math:`\mathbf{G}(\tilde{\mathbf{r}})=
    \begin{pmatrix} 
    \sin(\gamma |\tilde{r}|)\frac{\tilde{r}_x}{|\tilde{r}|}\\
    \sin(\gamma |\tilde{r}|)\frac{\tilde{r}_y}{|\tilde{r}|}\\
    \sin(\gamma |\tilde{r}|)\frac{\tilde{r}_z}{|\tilde{r}|}\\
    1+\cos(\gamma |\tilde{r}|) \\
    \end{pmatrix} \frac{\Theta(\tilde{r}_\mathrm{cutoff}-|\tilde{r}|)}{\gamma}`

    Parameters
    -----------
    lcs: (n_residues, 3, 3) array
        arrays of coordinates of the xyz unit vectors
    origins: (n_residues, 3) array
        arrays of origins
    cutoff: float
        unitless cutoff for eRMSD
    a: float
        re-scaling factor in x, y direction, usually it's 0.5 nm
    b: float
        re-scaling factor in z direction, usually it's 0.3 nm

    Returns
    ---------
    G: (n_residues, n_residues, 4) array

    """
    pairs = origins[:, np.newaxis] - origins[np.newaxis, :]
    # notice that this matrix is skew-symmetric
    # pairs[i, j] corresponds to origins[i] - origins[j]
    # which is vector base_j -> base_i.
    # Thus this needs to be evaluated in the ref system of base j
    # dimension (n_residues, n_residues, 3)

    r = np.einsum("ijk,jlk->ijl", pairs, lcs)
    # the dot product between vector pairs[i, j, k] and vector lcs[j, 0, k]
    # gives x values in the local reference system etc.

    # we need to introduce an anisotropic reduction
    r_tilde = np.einsum("ijk,k->ijk", r, np.array([1 / a, 1 / a, 1 / b]))

    r_tilde_norm = np.linalg.norm(r_tilde, axis=-1)
    gamma = np.pi / cutoff
    inverse_r_tilde_norm = np.where(r_tilde_norm != 0, 1 / r_tilde_norm, 0)
    G123 = np.einsum("ijk,ij->ijk", r_tilde, np.sin(gamma * r_tilde_norm) * inverse_r_tilde_norm)
    G4 = 1 + np.cos(gamma * r_tilde_norm)
    G = np.concatenate((G123, np.expand_dims(G4, axis=-1)), axis=-1)
    G = np.einsum(
        "ijk,ij->ijk", G, np.heaviside(cutoff - r_tilde_norm, np.zeros(r_tilde_norm.shape)) / gamma
    )

    return G


def ermsd(rs, reference, cutoff, a, b):
    r"""
    compute the eRMSD given the current snapshots and the reference coordinates
    Mathematical details can be found in
    [S. Bottaro, F. Di Palma,
    and G. Bussi, NAR, 2014](https://doi.org/10.1093/nar/gku972)

    First calculate the local coordinates based on C2/4/6 positions, and then
    calculate the G vectors.
    At last we sum up the square differences in G vectors.

    :math:`\varepsilon RMSD = \sqrt{\frac{1}{N} \sum_{j, k}
    |\mathbf{G}(\tilde{\mathbf{r}}_{jk}^\alpha) -
    \mathbf{G}(\tilde{\mathbf{r}}_{jk}^\beta)|^2}`

    Parameters
    ----------
    rs:
        (n_residues*3, 3) array, positions of the selected C2/4/6 atoms
    reference:
        (n_residues*3, 3) array, reference positions
    """
    lcs, origins = local_coordinates(rs)
    lcs_ref, origins_ref = local_coordinates(reference)
    N_res = origins.shape[0]
    Gs = g_vector(lcs, origins, cutoff, a, b)
    Gs_ref = g_vector(lcs_ref, origins_ref, cutoff, a, b)

    return np.sqrt(np.sum((Gs - Gs_ref) ** 2) / N_res)
