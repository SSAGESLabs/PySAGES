# funnel functions
from functools import partial

import jax.numpy as np
from jax import jit, grad
from jax.numpy import linalg

def distance(r, cell_size):
    diff = r[1:] - r[0]
    #diff = diff - np.round(diff / cell_size) * cell_size
    return np.linalg.norm(diff,axis=1)

def coordnum_energy(r, cell_size, c_mins, k, idx = np.asarray([ [0,1],[0,2],[0,3],[1,2],[1,3],[2,3] ])):
    total = 0 #all O somewhat close to C
    c_min = c_mins[0]
    dist = distance(r[:5],cell_size)
    total += 0.5*(np.where(dist > c_min, dist - c_min, 0.0)**2).sum()

    c_min = c_mins[1] #Os are all close to each other
    rO = r[1:5]
    dists = np.linalg.norm( rO[idx[:,0]] - rO[idx[:,1]],axis=1 )
    total += 0.5*(np.where(dists > c_min, dists - c_min, 0.0)**2).sum()
    return k * total


def intermediate_funnel(pos, ids, indexes, cell_size, c_mins, k):
    r = pos[ids[indexes]]
    return coordnum_energy(r, cell_size, c_mins, k)

def log_funnel():
    return 0.0

def external_funnel(data, indexes, cell_size, c_mins, k):
    pos = data.positions[:, :3]
    ids = data.indices
    bias = grad(intermediate_funnel)(pos, ids, indexes, cell_size, c_mins, k)
    proj = log_funnel()
    return bias, proj

def get_funnel_force(indexes, cell_size, c_mins, k):
    funnel_force = partial(
        external_funnel,
        indexes=indexes, cell_size=cell_size, c_mins=c_mins, k=k)
    return jit(funnel_force)
