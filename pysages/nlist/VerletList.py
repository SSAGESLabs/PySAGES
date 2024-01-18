import jax
import jax.numpy as np
from jax import lax

class VerletList:
    """
    Verlet list neighbor list algorithm for 3D systems implemented in JAX. Originally implemented to work with the CellList class as a hybrid neighbor list algorithm.

    Returns:
        VerletList: VerletList object containing the following attributes:

            cutoff (float): cutoff distance for neighbor list (scalar)
            buffer_size (int): max number of neighbors per cell (scalar)

    """
    cutoff: float
    buffer_size: int

    def __init__(self, cutoff: float) -> None:
        self.cutoff = lax.stop_gradient(cutoff) # scalar
        self.buffer_size = None # scalar
    
    def set_buffer_size(self, buffer_size: int) -> None:
        """
        Setter for buffer_size attribute.

        Args:
            buffer_size (int): max number of neighbors per cell

        Returns:
            None
        """
        self.buffer_size = buffer_size
        return None

    def _get_dist(self, i: jax.Array, j: jax.Array) -> np.float32:
        """
        Calculate the distance between two particles. (helper function for _pairwise_dist to use with vmap)

        Args:
            i (jax.Array): Position of particle i (3, )
            j (jax.Array): Position of particle j (3, )

        Returns:
            np.float32: Distance between particles i and j (scalar)
        """
        return np.linalg.norm(i-j)

    def _pairwise_dist(self, pos: jax.Array, ref: jax.Array) -> jax.Array:
        """
        Calculate the pairwise distance between particles in pos and a single reference particle. (helper function for get_neighbor_ids to use with vmap)

        Args:
            pos (jax.Array): position of particles (N, 3)
            ref (jax.Array): position of reference particle (3, )

        Returns:
            jax.Array: array of distances between particles and reference particle (N, )
        """
        return jax.vmap(self._get_dist, (0, None))(pos, ref)

    def _is_neighbor(self, dist: jax.Array) -> jax.Array:
        """
        Check if a particle is a neighbor of the reference particle. (helper function for get_neighbor_ids to use with vmap)

        Args:
            dist (jax.Array): Array of distances between particles and reference particle (N, )

        Returns:
            jax.Array: Array of bools indicating whether a particle is a neighbor of the reference particle (N, )
        """
        return dist < self.cutoff
    
    def get_neighbor_ids(self, pos: jax.Array, sparse: bool = False, mask_self: bool = False) -> jax.Array:
        """
        Get neighbor ids for each particle in pos matrix.

        Args:
            pos (jax.Array): Array of particle positions (N, 3)
            sparse (bool, optional): Whether to return the full (N, N) matrix of neighborhood or an Array. Defaults to False.
            mask_self (bool, optional): Whether to exclude self from neighbor list. Defaults to False.

        Returns:
            jax.Array: Array of neighbor ids for each particle (N, ) or (N, N) matrix of bools indicating whether a particle is a neighbor of another particle.
        """
        # if buffer_size is not set, set it to the number of particles
        if self.buffer_size is None:
            self.buffer_size = pos.shape[0]
        # calculate the pairwise distances between all particles
        pair_dists = jax.vmap(self._pairwise_dist, (None, 0))(pos, pos)
        # check if a particle is a neighbor of another particle based on the cutoff distance
        is_neighbor = jax.vmap(self._is_neighbor)(pair_dists)
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
    
    def get_neighborhood(self, pos: jax.Array, ref: jax.Array) -> jax.Array:
        """
        Get the neighborhood of a specific particle. Implemented to work with the CellList class as a hybrid neighbor list algorithm.

        Args:
            pos (jax.Array): Array of particle positions (N, 3)
            ref (jax.Array): position of reference particle (3, )

        Returns:
            jax.Array: Array of bools indicating whether a particle is a neighbor of the reference particle (N, )
        """
        # calculate the pairwise distances between all particles and the reference particle
        pair_dists = self._pairwise_dist(pos, ref)
        # check if a particle is a neighbor of the reference particle based on the cutoff distance
        is_neighbor = self._is_neighbor(pair_dists)
        return is_neighbor