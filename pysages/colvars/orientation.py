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
        Must be a list of indices with dimension (n_nucleotides*3,).
        The order is tricky,
        for Purines (A, G) the order in the tuple should be C2, C6, C4;
        for Pyrimidines (U, C) the order in the tuple should be C2, C4, C6.
        This weird order is simply due to the numbering of the base.
        Please make sure the order is correct,
        otherwise the results will be wrong.
    references: list[tuple(float)]
        Must be a list of tuple of floats, with dimension (n_nucleotides*3, 3)
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


def calc_local_reference_systems(rs):
    """
    calculate the local coordinate system for eRMSD calculation:
    origin and orthogonal x/y/z axis in the six membered rings.
    The x-axis is given by center of geometry (cog) -> C2,
    the z-axis is given by cross product
    between x-axis and cog -> second atom (C2 for U/C, C4 for A/G),
    and y-axis is given by cross product of z-axis and x-axis.

    Parameters
    ----------
    rs: (n_nucleotides, 3, 3) array
        positions of C2/4/6 atoms of every bases. The last dimension is x/y/z.
        The order is C2, C6, C4 for A/G and C2, C4, C6 for U/C.

    Returns
    ----------
    local_reference_systems: (n_nucleotides, 3, 3) array
        x, y, z unit vectors of each base

    origins: (n_nucleotides, 3) array
        origins (center of geometry) of the six membered rings of each base

    """
    origins = np.average(rs, axis=1)
    # Cog->C2 (x axis)
    x = rs[:, 0] - origins
    x_hat = np.einsum("ij,i->ij", x, 1 / np.linalg.norm(x, axis=-1))
    # Cog->C4/C6
    cog_2nd_atom = rs[:, 1] - origins
    # z, y axis
    z = np.cross(x_hat, cog_2nd_atom, axis=1)
    z_hat = np.einsum("ij,i->ij", z, 1 / np.linalg.norm(z, axis=-1))

    y = np.cross(z_hat, x_hat, axis=1)
    y_hat = np.einsum("ij,i->ij", y, 1 / np.linalg.norm(y, axis=1))
    local_reference_systems = np.stack((x_hat, y_hat, z_hat), axis=1)

    return local_reference_systems, origins


def g_vector(local_reference_systems, origins, cutoff, a, b):
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
    local_reference_systems: (n_nucleotides, 3, 3) array
        arrays of coordinates of the xyz unit vectors
    origins: (n_nucleotides, 3) array
        arrays of origins
    cutoff: float
        unitless cutoff for eRMSD
    a: float
        re-scaling factor in x, y direction, usually it's 0.5 nm
    b: float
        re-scaling factor in z direction, usually it's 0.3 nm

    Returns
    ---------
    G: (n_nucleotides, n_nucleotides, 4) array

    """
    pairs = origins[:, np.newaxis] - origins[np.newaxis, :]
    # notice that this matrix is skew-symmetric
    # pairs[i, j] corresponds to origins[i] - origins[j]
    # which is vector base_j -> base_i.
    # Thus this needs to be evaluated in the ref system of base j
    # dimension (n_nucleotides, n_nucleotides, 3)

    r = np.einsum("ijk,jlk->ijl", pairs, local_reference_systems)
    # the dot product between vector pairs[i, j, k]
    # and vector local_reference_systems[j, 0, k]
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


def reshape_coordinates(rs, reference):
    """
    reshape the coordinates so that they have easier dimensions to work with.
    In pysages, we have been concatenating the nucleotides
    with sites in each base.
    E.g [ C2, C4, C6, C2, C6, C4] for a 2nt RNA with sequence: UA.
    we want to reshape the coordinates so that it's
    [[C2, C4, C6], [C2, C6, C4]]

    Parameters
    -------------
    rs: (n_nucleotides*3, 3) array
    reference: (n_nucleotides*3, 3) array

    Returns
    -------------
    rs: (n_nucleotides, 3, 3) array
    reference: (n_nucleotides, 3, 3) array
    """
    n_sites, n_xyz = rs.shape
    n_sites_per_base = 3
    assert n_xyz == 3, f"the number of coordinate dimension is {n_xyz}" " but should be 3!"
    assert n_sites % n_sites_per_base == 0, (
        f"Please make sure the indices are in dimension " f" 3 * n_nucleotides! Now it's {n_sites}."
    )
    n_nucleotides = int(n_sites / n_sites_per_base)
    rs = rs.reshape(n_nucleotides, n_sites_per_base, n_xyz)
    reference = reference.reshape(n_nucleotides, n_sites_per_base, n_xyz)

    return rs, reference


def ermsd_core(rs, reference, cutoff, a, b):
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
        (n_nucleotides, 3, 3) array, positions of the selected C2/4/6 atoms
    reference:
        (n_nucleotides, 3, 3) array, reference positions

    Return
    -----------
    eRMSD: float
        value of the eRMSD
    """
    system_and_origins = calc_local_reference_systems(rs)
    local_reference_systems, origins = system_and_origins
    system_and_origins_ref = calc_local_reference_systems(reference)
    local_reference_systems_ref, origins_ref = system_and_origins_ref
    N_res = origins.shape[0]
    Gs = g_vector(local_reference_systems, origins, cutoff, a, b)
    Gs_ref = g_vector(local_reference_systems_ref, origins_ref, cutoff, a, b)

    return np.sqrt(np.sum((Gs - Gs_ref) ** 2) / N_res)


def ermsd(rs, reference, cutoff, a, b):
    """
    compute the eRMSD between the current snapshot and the reference

    Parameters
    ----------
    rs:
        (n_nucleotides*3, 3) array, positions of the selected C2/4/6 atoms
    reference:
        (n_nucleotides*3, 3) array, reference positions

    Return
    -----------
    eRMSD: float
        value of the eRMSD
    """
    rs, reference = reshape_coordinates(rs, reference)
    return ermsd_core(rs, reference, cutoff, a, b)


@multicomponent
class ERMSDCG(CollectiveVariable):
    """
    Use a reference to calculate the eRMSD of
    a set of Coarse-grained RNA structures.
    Mathematical details can be found in
    [S. Bottaro, F. Di Palma,
    and G. Bussi, NAR, 2014](https://doi.org/10.1093/nar/gku972)

    To use this method, we assume the model has access to 3 sites in the base.
    The method will use the ideal base geometry to reconstruct C2/C4/C6
    that's needed for eRMSD.
    The known base sites are assumed to be RACER-like mapping:
        A: C8, N6, C2
        U: C6, O4, O2
        G: C8, O6, N2
        C: C6, N4, O2
    Will refer to these mapped sites as B1/B2/B3

    Parameters
    ------------
    indices: list[int].
        Must be a list of indices with dimension (n_nucleotides*3,).
        The order is tricky.
        A: C8, N6, C2
        U: C6, O4, O2
        G: C8, O6, N2
        C: C6, N4, O2
        Please make sure the order is correct,
        otherwise the results will be wrong.
    reference: list[tuple(float)]
        Must be a list of tuple of floats, with dimension (n_nucleotides*3, 3)
        The last dimension is the x,y,z of the
        cartesian coordinates of the reference position of the selected sites
        with the same order as the indices.
    sequence: list[int]
        a list of int with length=n_nucleotides, each element is 0 (A) or 1 (U)
        or 2 (G) or 3 (C)
    local_coordinates: (4,3,3) array
        Must be a list of tuple of floats, with dimension (4, 3, 3)
        Local coordinates of [C2, C6, C4] (A/G) or [C2, C4, C6] (U/C).
        The reference system is given by this:
            x-axis is unit vector, pointing from the center of B1/B2/B3 to B1
            z-axis is cross-product between B1->B2, B1->B3, normalized
            y-axis is cross-product between z-axis and x-axis.
        The axis 0 of the local_coordinates is A/U/G/C.
        The axis 1 is [C2, C6, C4] (A/G) or [C2, C4, C6] (U/C).
        The axis 2 is coefficient for x-axis, y-axis and z-axis
        (Coefficient for z-axis is 0
        because sites in the base share the same plane)
        The default value is local coordinates
        from RACER like mapping B1/B2/B3
        to SPQR type of mapping. Notice the unit is in nm.
    cutoff: float
        unitless cutoff. Default is 2.4
    a: float
        re-scaling scale in x and y direction. Default is 0.5 (nm).
    b: float
        re-scaling scale in z direction. Default is 0.3 (nm).
    """

    def __init__(
        self,
        indices,
        reference,
        sequence,
        local_coordinates=[
            [
                [-0.1333021, -0.1762476, -0.0000001],
                [-0.0857174, 0.0499809, -0.0000033],
                [0.0795015, -0.1210861, -0.0001387],
            ],
            [
                [-0.0253117, -0.1224880, 0.0003341],
                [-0.0236469, 0.1239022, -0.0006548],
                [0.1811280, 0.0000000, -0.0000000],
            ],
            [
                [-0.1167570, -0.1372227, -0.0002264],
                [-0.0409667, 0.0959927, -0.0012938],
                [0.1013132, -0.0983306, 0.0005419],
            ],
            [
                [-0.0267224, -0.1188717, 0.0004939],
                [-0.0254680, 0.1138277, 0.0008671],
                [0.1815304, -0.0000000, -0.0000000],
            ],
        ],
        cutoff=2.4,
        a=0.5,
        b=0.3,
    ):
        super().__init__(indices)
        self.reference = np.asarray(reference)
        self.sequence = np.asarray(sequence)
        self.local_coordinates = np.asarray(local_coordinates)
        self.cutoff = cutoff
        self.a = a
        self.b = b

    @property
    def function(self):
        return lambda r: ermsd_cg(
            r, self.reference, self.sequence, self.local_coordinates, self.cutoff, self.a, self.b
        )


def infer_base_coordinates(local_reference_systems, origins, sequence, local_coordinates):
    """
    Infer the coordinates of C2/C4/C6 based on the local reference system
    These are needed for eRMSD calculation.

    Parameters
    ------------
    rs: (n_nucleotides, 3, 3) array
        current coordinates of selected sites of the base
    sequence: (n_nucleotides, ) array
        an array of int with length=n_nucleotides,
        each element is 0 (A) or 1 (U) or 2 (G) or 3 (C)
    local_coordinates: (4,3,2) array
        Must be a list of tuple of floats, with dimension (4, 3, 2)
        Local coordinates of [C2, C6, C4] (A/G) or [C2, C4, C6] (U/C).
        The reference system is given by this:
            x-axis is unit vector, pointing from the center of B1/B2/B3 to B1
            z-axis is cross-product between B1->B2, B1->B3, normalized
            y-axis is cross-product between z-axis and x-axis.
        The axis 0 of the local_coordinates is A/U/G/C
        The axis 1 is [C2, C6, C4] (A/G) or [C2, C4, C6] (U/C).
        The axis 2 is coefficient for x-axis, y-axis and z-axis
        (coefficient for z-axis is close to 0
        because sites in the base are almost in the same plane)
        The default value is local coordinates
        from RACER like mapping B1/B2/B3
        to SPQR type of mapping.

    Return
    -------------
    C246_coords: (n_nucleotides, 3, 3) array
        coordinates of C2/4/6 in correct orders, ready for eRMSD calculation
    """

    C246_coords = np.einsum(
        "jlk,jml->jmk", local_reference_systems[:, :, :], local_coordinates[sequence]
    )
    C246_coords += origins[:, np.newaxis, :]

    return C246_coords


def ermsd_cg(rs, reference, sequence, local_coordinates, cutoff, a, b):
    """
    coarse-grained version of eRMSD which pass in different base sites
    other than the default C2/4/6 used by eRMSD.
    Thus we first need to infer the positions of C2/4/6 using rigid
    ideal base references and then do standard eRMSD calculations.

    Parameters
    ------------
    rs: (n_nucleotides*3, 3) array
        current snapshots of selected sites
    reference: (n_nucleotides*3, 3) array
        reference of the selected sites
        The last dimension is the x,y,z of the
        cartesian coordinates of the reference position of the C2/C4/C6 atoms,
        with the same order as the indices.
    sequence: list[int]
        a list of int with length=n_nucleotides, each element is 0 (A) or 1 (U)
        or 2 (G) or 3 (C)
    local_coordinates: (4,3,2) array
        Must be a list of tuple of floats, with dimension (4, 3, 2)
        Local coordinates of [C2, C6, C4] (A/G) or [C2, C4, C6] (U/C).
    cutoff: float
        unitless cutoff. Default is 2.4
    a: float
        re-scaling scale in x and y direction. Default is 0.5 (nm).
    b: float
        re-scaling scale in z direction. Default is 0.3 (nm).
    """
    rs, reference = reshape_coordinates(rs, reference)
    system_and_origins = calc_local_reference_systems(rs)
    local_reference_systems, origins = system_and_origins
    # notice that the function calc_local_reference_systems
    # exactly works for different mappings
    # please note that these lcs are very different from lcs in eRMSD
    # simply because the input sites are chosen differently.
    system_and_origins_ref = calc_local_reference_systems(reference)
    local_reference_systems_ref, origins_ref = system_and_origins_ref

    # infer C2/4/6
    C246_coords = infer_base_coordinates(
        local_reference_systems, origins, sequence, local_coordinates
    )
    C246_coords_ref = infer_base_coordinates(
        local_reference_systems_ref, origins_ref, sequence, local_coordinates
    )
    return ermsd_core(C246_coords, C246_coords_ref, cutoff, a, b)
