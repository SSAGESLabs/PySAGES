import jax
from jax import jit
import jax.numpy as np

@jit
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

@jit
def _tuple_to_idx(tup: jax.Array, cell_edge: jax.Array) -> np.int32 | jax.Array:
    """
    Covnert cell index from tuple to scalar.

    Args:
        tup (jax.Array): [index in x, index in y, index in z]
        cell_edge (jax.Array): Number of cells in each dimension

    Returns:
        np.int32 or jax.Array: Scalar index of cell or (N, ) array of scalar indices
    """
    return np.int32(tup[...,0]*cell_edge[1]*cell_edge[2] + tup[...,1]*cell_edge[2] + tup[...,2])

@jit
def _get_cell_ids(pos: jax.Array, cell_cut: jax.Array, cell_edge: jax.Array) -> jax.Array:
    """
    Get scalar cell ids for each particle (row) in pos matrix.

    Args:
        pos (jax.Array): matrix of particle positions (N, 3)
        cell_cut (jax.Array): Cut off distance for each cell
        cell_edge (jax.Array): Number of cells in each dimension

    Returns:
        jax.Array: Array of cell ids for each particle (N, )
    """
    cell_tuples: np.int32 = pos//cell_cut
    cell_ids = _tuple_to_idx(cell_tuples, cell_edge)
    return cell_ids

@jit
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

@jit
def _get_neighbor(ids: jax.Array, cell_edge: jax.Array) -> jax.Array:
    """
    Wrap the tuple cell ids of particles for each neighbor. (helper function for _get_neighbor_ids to use with vmap)

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
