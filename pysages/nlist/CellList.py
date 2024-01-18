from typing import Tuple


import jax
from jax import numpy as np
from jax import lax
from cellfuncs import _idx_to_tuple, _tuple_to_idx, _get_cell_ids, _get_neighbor

class CellList:
    """
    Cell list neighbor list algorithm for 3D systems implemented in JAX.
    Loosely based on https://aiichironakano.github.io/cs596/01-1LinkedListCell.pdf

    Raises:
        ValueError: If the cell list is not initialized before calling get_neighbor_ids()

    Returns:
        CellList: CellList object containing the following attributes:

            box (Tuple): box size (3, )
            cutoff (float): cutoff distance for neighbor list (scalar)
            cell_edge (jax.Array): number of cells in each dimension (3, )
            cell_cut (jax.Array): cell size in each dimension (3, )
            cell_num (int): total number of cells (scalar)
            cell_idx (jax.Array): cell index for each particle (N, )
            buffer_size_cell (int): max number of neighbors per cell (scalar). If not set, it is set to 50% larger than the average number of particles per cell.
    """
    box: Tuple
    cutoff: float
    cell_edge: jax.Array
    cell_cut: jax.Array
    cell_num: int
    cell_idx: jax.Array
    buffer_size_box: int

    def __init__(self, box: Tuple, cutoff: float, buffer_size_cell: int = None) -> None:
        self.box = lax.stop_gradient(np.asarray(box)) # (3, )
        self.cutoff = lax.stop_gradient(cutoff) # scalar
        self.cell_edge = np.floor(self.box/self.cutoff) # (3, )
        self.cell_cut = self.box/self.cell_edge # (3, )
        self.cell_num = np.prod(self.cell_edge) # scalar
        self.cell_idx = None # (N, )
        self.buffer_size_cell = buffer_size_cell # scalar
    
    def set_buffer_size(self, buffer_size_cell: int) -> None:
        """
        Setter for buffer_size_cell attribute.

        Args:
            buffer_size_cell (int): max number of neighbors per cell

        Returns:
            None
        """
        self.buffer_size_cell = buffer_size_cell
        return None
    
    def _get_neighbor_ids(self, idx: int) -> jax.Array:  
        """
        Get neighbor ids for a single particle.

        Args:
            idx (int): index of the particle in the pos matrix (scalar)

        Raises:
            ValueError: If the neighbor list overflows
        
        Returns:
            jax.Array: Array of neighbor ids for the particle (N, )
        """
        cell_id = self.cell_idx[idx] # index of the cell that the particle is in scalar
        cell_id = np.expand_dims(cell_id, axis=0) # scalar to (1, )
        
        cell_tuple = _idx_to_tuple(cell_id, self.cell_edge) # tuple of the cell that the particle is in (1, dim)
        
        neighbor_tuples = []
        for i in [-1, 0, 1]: # loop over cells behind and ahead of the current cell in each dimension
            for j in [-1, 0, 1]: 
                    for k in [-1, 0, 1]:
                        neighbor_tuples.append(np.asarray([cell_tuple[0]+i, cell_tuple[1]+j, cell_tuple[2]+k]))
        
        neighbor_tuples = np.asarray(neighbor_tuples) # list to jax.Array (27, dim)
        neighbor_tuples_wrapped = jax.vmap(_get_neighbor, in_axes=(0, None), out_axes=0)(neighbor_tuples, self.cell_edge) # wrap the cell ids of the neighbors (27, dim)
        
        # get scalar ids for the neighboring cells
        neighbor_cell_ids = jax.vmap(_tuple_to_idx, (0, None))(neighbor_tuples_wrapped, self.cell_edge)
        
        neighbor_ids = [] # get ids of the particles in the neighboring cells. -1 is used as a filler for empty cells.
        for cidx in neighbor_cell_ids:
            neighbor_ids.append(np.where(self.cell_idx == cidx, fill_value=-1, size=self.buffer_size_cell)[0])

            
        # concatenate the neighbor ids into a single array.
        neighbor_ids = np.concatenate(neighbor_ids, axis=-1)
        return neighbor_ids
    
    def get_neighbor_ids(self, idxs: jax.Array, mask_self: bool= False) -> jax.Array:
        """
        Get neighbor ids for a list of particles. Uses vmap to vectorize the _get_neighbor_ids function.

        Args:
            idxs (jax.Array): Array of particle indices (n, )
            mask_self (bool, optional): Whether to exclude self from neighbor list. Defaults to False.

        Raises:
            ValueError: If the cell list is not initialized before calling get_neighbor_ids()

        Returns:
            jax.Array: Array of neighbor ids for each particle (n, buffer_size_nl))
        """
        if self.cell_idx is None:
            raise ValueError("Cell list is not initialized. Call get_cell_ids() first.")
        
        # convert idxs to jax.Array if it is not already
        if not isinstance(idxs, np.ndarray):
            idxs = np.asarray(idxs)
        # expand dims if idxs is a single particle
        if len(idxs.shape) == 0:
            idxs = np.expand_dims(idxs, axis=-1)

        if idxs.shape[0] == 1: # single particle case, no vmap
            # get neighbor ids for the particle
            n_ids = self._get_neighbor_ids(idxs[0])
            # check for overflow
            min_buffer = np.count_nonzero(n_ids == -1, axis=-1)
            if min_buffer < 27: # if there are less than 27 -1s in a row of the neighbor list, there is an overflow from buffer_size_cell
                raise ValueError("Neighbor list overflow. Increase buffer_size_cell.")
            # remove self from neighbor list if mask_self is True
            if mask_self:
                n_ids = n_ids[n_ids != idxs[0]]
            # sort 
            n_ids = np.sort(n_ids)[::-1]
            # truncate. Remove the -1s from the end of the neighbor list(smallest possible neighbor list).
            n_ids = n_ids[:-min_buffer]
            return n_ids
        else:
            # get neighbor ids for the particles
            n_ids = jax.vmap(self._get_neighbor_ids)(idxs)
            # check for overflow
            min_buffer = np.min(np.count_nonzero(n_ids == -1, axis=-1))
            if min_buffer < 27: # if there are less than 27 -1s in a row of the neighbor list, there is an overflow from buffer_size_cell
                raise ValueError("Neighbor list overflow. Increase buffer_size_cell.")
            # remove self from neighbor list if mask_self is True
            if mask_self:
                # set the self index to -1
                n_ids = n_ids.at[..., n_ids == idxs[:, None]].set(-1)
                # add one to the minimum buffer size to account for the -1 just added
                min_buffer += 1
            # sort
            n_ids = np.sort(n_ids, axis=-1)[:, ::-1]
            # truncate. Remove the -1s from the end of the neighbor list so that the row with the least -1s will have none (smallest possible neighbor list).
            n_ids = n_ids[:, :-min_buffer]

        return n_ids
    
    def initiate(self, pos: jax.Array) -> None:
        """
        Initialize the cell list.

        Args:
            pos (jax.Array): Array of particle positions (N, 3)

        Returns:
            None
        """
        if self.buffer_size_cell is None: # can be set manually if needed
            self.buffer_size_cell = np.int32(np.ceil(pos.shape[0]//self.cell_num * 1.5)) # set the size of the nl list per cell to 50% larger than the average number of particles per cell
        # get the cell ids
        self.cell_idx = np.zeros(pos.shape[0], dtype=np.int32)
        self.cell_idx = _get_cell_ids(pos, self.cell_cut, self.cell_edge)
        return None
    
    def update(self, pos: jax.Array) -> None:
        """
        Update the cell list.

        Args:
            pos (jax.Array): Array of particle positions (N, 3)

        Returns:
            None
        """
        # update the cell ids by calling _get_cell_ids
        self.cell_idx = _get_cell_ids(pos, self.cell_cut, self.cell_edge)
        return None
