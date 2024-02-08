import jax
import jax.numpy as np

def _get_dist(i: jax.Array, j: jax.Array, box_size: jax.Array) -> np.float32:
        """
        Calculate the distance between two particles. (helper function for _pairwise_dist to use with vmap)

        Args:
            i (jax.Array): Position of particle i (3, )
            j (jax.Array): Position of particle j (3, )

        Returns:
            np.float32: Distance between particles i and j (scalar)
        """
        dx = i[0] - j[0]
        dx = np.where(dx > box_size[0]/2, dx - box_size[0], dx)
        dy = i[1] - j[1]
        dy = np.where(dy > box_size[1]/2, dy - box_size[1], dy)
        dz = i[2] - j[2]
        dz = np.where(dz > box_size[2]/2, dz - box_size[2], dz)
        return np.sqrt(dx**2 + dy**2 + dz**2)

def _pairwise_dist(pos: jax.Array, ref: jax.Array, box_size: jax.Array) -> jax.Array:
    """
    Calculate the pairwise distance between particles in pos and a single reference particle. (helper function for get_neighbor_ids to use with vmap)

    Args:
        pos (jax.Array): position of particles (N, 3)
        ref (jax.Array): position of reference particle (3, )

    Returns:
        jax.Array: array of distances between particles and reference particle (N, )
    """
    return jax.vmap(_get_dist, (0, None, None))(pos, ref, box_size)

def _is_neighbor(dist: jax.Array, cutoff: float) -> jax.Array:
    """
    Check if a particle is a neighbor of the reference particle. (helper function for get_neighbor_ids to use with vmap)

    Args:
        dist (jax.Array): Array of distances between particles and reference particle (N, )
        cutoff (float): cutoff distance for neighbor list (scalar)

    Returns:
        jax.Array: Array of bools indicating whether a particle is a neighbor of the reference particle (N, )
    """
    return dist <= cutoff

def get_neighbor_ids(pos: jax.Array, cutoff: float, box_size: jax.Array, sparse: bool = False, mask_self: bool = False) -> jax.Array:
    """
    Get neighbor ids for each particle in pos matrix.

    Args:
        pos (jax.Array): Array of particle positions (N, 3)
        sparse (bool, optional): Whether to return the full (N, N) matrix of neighborhood or an Array. Defaults to False.
        mask_self (bool, optional): Whether to exclude self from neighbor list. Defaults to False.

    Returns:
        jax.Array: Array of neighbor ids for each particle (N, ) or (N, N) matrix of bools indicating whether a particle is a neighbor of another particle.
    """
    # calculate the pairwise distances between all particles
    pair_dists = jax.vmap(_pairwise_dist, (None, 0, None))(pos, pos, box_size)
    # check if a particle is a neighbor of another particle based on the cutoff distance
    is_neighbor = jax.vmap(_is_neighbor, (0, None))(pair_dists, cutoff)
    # remove self from neighbor list if mask_self is True
    if mask_self:
        i, j = np.diag_indices(is_neighbor.shape[0])
        is_neighbor = is_neighbor.at[..., i, j].set(False)
    # return a list of arrays if sparse is True
    if sparse: # return a list of arrays
        neighbor_list = []
        for row in is_neighbor:
            neighbor_list.append(np.where(row)[0])
        return neighbor_list
    
    return is_neighbor # return a NxN array of bools

def get_neighborhood(pos: jax.Array, ref: jax.Array, cutoff: float, box_size: jax.Array) -> jax.Array:
    """
    Get the neighborhood of a reference particle.

    Args:
        pos (jax.Array): Array of particle positions (N, 3)
        ref (jax.Array): position of reference particle (3, )

    Returns:
        jax.Array: Array of bools indicating whether a particle is a neighbor of the reference particle (N, )
    """
    # calculate the pairwise distances between all particles and the reference particle
    pair_dists = _pairwise_dist(pos, ref, box_size)
    # check if a particle is a neighbor of the reference particle based on the cutoff distance
    is_neighbor = _is_neighbor(pair_dists, cutoff)
    return is_neighbor