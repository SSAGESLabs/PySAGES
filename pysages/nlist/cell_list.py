from typing import Union


import jax
from jax import numpy as np

def _tuple_to_idx(tup: jax.Array, cell_edge: jax.Array) -> Union[np.int32, jax.Array]:
    """
    Covnert cell index from tuple to scalar.

    Args:
        tup (jax.Array): [index in x, index in y, index in z]
        cell_edge (jax.Array): Number of cells in each dimension

    Returns:
        np.int32 or jax.Array: Scalar index of cell or (N, ) array of scalar indices
    """
    return np.int32(tup[...,0]*cell_edge[1]*cell_edge[2] + tup[...,1]*cell_edge[2] + tup[...,2])

def _idx_to_tuple(idx: int, cell_edge: jax.Array) -> jax.Array:
    """
    Convert cell index from scalar to tuple.

    Args:
        idx (int): Scalar index of cell
        cell_edge (jax.Array): Number of cells in each dimension

    Returns:
        jax.Array: [index in x, index in y, index in z] (3,) or (N, 3)
    """
    x: np.int32 = idx//(cell_edge[1]*cell_edge[2])
    y: np.int32 = (idx//cell_edge[2])%cell_edge[1]
    z: np.int32 = idx%cell_edge[2]
    return np.concatenate([x, y, z], axis=-1, dtype=np.int32)

def get_cell_list(pos: jax.Array, box_size: jax.Array, cutoff: float) -> jax.Array:
    """
        Initialize the cell list.

        Args:
            pos (jax.Array): Array of particle positions (N, 3)
            box_size (Tuple): box size (3, )
            cutoff (float): cutoff distance for neighbor list (scalar)

        Returns:
            cell_idx (jax.Array): cell index for each particle (N, )
        """
    #setup the box parameters
    cell_edge = np.floor(box_size/cutoff) # (3, )
    cell_cut = box_size/cell_edge # (3, )
    # get the cell ids
    cell_tuples = pos//cell_cut
    cell_idx = _tuple_to_idx(cell_tuples, cell_edge)
    return cell_idx

def _wrap_cell_ids(cell_ids: jax.Array, cell_edge: np.int32) -> jax.Array:
    """
    Wraps the cell ids of particles in edge cells. (single dimension)

    Args:
        cell_ids (jax.Array): Array of tuple cell ids in the current dimension for each particle (N, 1)
        cell_edge (np.int32): Number of cells in current dimension

    Returns:
        jax.Array: Wrapped cell ids (tuple) for each particle (N, 3)
    """
    out_of_bound_low = (cell_ids == -1) # if cell id is -1 (out of bound from below)
    out_of_bound_high = (cell_ids == cell_edge) # if cell id equal to the number of cells in that dimension (out of bound from above)
    cell_ids = np.where(out_of_bound_low, cell_edge-1, cell_ids) # if out of bound, then wrap around from below
    cell_ids = np.where(out_of_bound_high, 0, cell_ids) # if out of bound, then wrap around from above
    return cell_ids

def _get_neighbor_box(ids: jax.Array, cell_edge: jax.Array) -> jax.Array:
    """
    Wrap the tuple cell ids of particles for each neighbor. (helper function for get_neighbor_ids to use with vmap)

    Args:
        ids (jax.Array): Array of tuple cell ids (N, 3)
        cell_edge (jax.Array): Array of number of cells in each dimension (3, )

    Returns:
        jax.Array: Wrapped tuple cell ids (N, 3)
    """
    i, j, k = ids
    x = _wrap_cell_ids(i, cell_edge[0])
    y = _wrap_cell_ids(j, cell_edge[1])
    z = _wrap_cell_ids(k, cell_edge[2])
    return np.asarray([x, y, z])

def get_neighbor_ids(box_size: jax.Array, cutoff: float, cell_idx: jax.Array, idx: int, buffer_size_cell: int) -> jax.Array:  
        """
        Get neighbor ids for a single particle.

        Args:
            box_size (Tuple): box size (3, )    
            cutoff (float): cutoff distance for neighbor list (scalar)
            cell_idx (jax.Array): cell index for each particle (N, )
            idx (int): index of the particle in the pos matrix (scalar)
            buffer_size_cell (int): buffer size for the cell list (scalar)

        Raises:
            ValueError: If the neighbor list overflows
        
        Returns:
            jax.Array: Array of neighbor ids for the particle (N, )
        """
        cell_edge = np.floor(box_size/cutoff) # (3, )
        cell_id = cell_idx[idx] # index of the cell that the particle is in scalar
        cell_id = np.expand_dims(cell_id, axis=0) # scalar to (1, )
        
        cell_tuple = _idx_to_tuple(cell_id, cell_edge) # tuple of the cell that the particle is in (1, dim)
        
        neighbor_tuples = []
        for i in [-1, 0, 1]: # loop over cells behind and ahead of the current cell in each dimension
            for j in [-1, 0, 1]: 
                    for k in [-1, 0, 1]:
                        neighbor_tuples.append(np.asarray([cell_tuple[0]+i, cell_tuple[1]+j, cell_tuple[2]+k]))
        
        neighbor_tuples = np.asarray(neighbor_tuples) # list to jax.Array (27, dim)
        neighbor_tuples_wrapped = jax.vmap(_get_neighbor_box, in_axes=(0, None), out_axes=0)(neighbor_tuples, cell_edge) # wrap the cell ids of the neighbors (27, dim)
        
        # get scalar ids for the neighboring cells
        neighbor_cell_ids = jax.vmap(_tuple_to_idx, (0, None))(neighbor_tuples_wrapped, cell_edge)
        
        neighbor_ids = [] # get ids of the particles in the neighboring cells. -1 is used as a filler for empty cells.
        for cidx in neighbor_cell_ids:
            neighbor_ids.append(np.where(cell_idx == cidx, fill_value=-1, size=buffer_size_cell)[0])

            
        # concatenate the neighbor ids into a single array.
        neighbor_ids = np.concatenate(neighbor_ids, axis=-1)
        return neighbor_ids

def get_neighbors_list(box_size: jax.Array, cutoff: float, cell_idx: jax.Array, idxs: jax.Array, buffer_size_cell: int, mask_self: bool= False) -> jax.Array:
    """
    Get neighbor ids for a list of particles. Uses vmap to vectorize on get_neighbor_ids function.

    Args:
        cell_idx (jax.Array): cell index for each particle (N, )
        idxs (jax.Array): Array of particle indices (n, )
        mask_self (bool, optional): Whether to exclude self from neighbor list. Defaults to False.

    Raises:
        ValueError: If the cell list is not initialized before calling get_neighbor_ids()

    Returns:
        jax.Array: Array of neighbor ids for each particle (n, buffer_size_nl))
    """

    # get neighbor ids for the particles
    n_ids = jax.vmap(get_neighbor_ids, in_axes=(None, None, None, 0, None))(box_size, cutoff, cell_idx, idxs, buffer_size_cell)
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
