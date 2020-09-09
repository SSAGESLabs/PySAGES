# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


from jax.tree_util import register_pytree_node


# From:
# - https://github.com/google/jax/issues/446
# - https://github.com/google/jax/issues/806
def register_pytree_namedtuple(cls):
    register_pytree_node(
        cls,
        lambda xs: (tuple(xs), None),  # tell JAX how to unpack
        lambda _, xs: cls(*xs)         # tell JAX how to pack back
    )
    return cls


#def wrap_around(boxsize, r):
#    half_boxsize = boxsize / 2
#    return np.mod(r + half_boxsize, boxsize) - half_boxsize
