import jax
from jax import numpy as np
from functools import partial
from pysages.colvars.core import CollectiveVariable
from .train_committor_dist import CommittorNN_Dist_Lip, make_forward_eval, CommittorNN_PIV, CommittorNN_PIV_shiftsig
from pysages.typing import JaxArray, List, Sequence
from flax import serialization

class Committor_CV_dist_lipschitz(CollectiveVariable):
    def __init__(self, indices: List, params_path: str, tri_idx1: JaxArray, tri_idx2: JaxArray):
        super().__init__(indices)

        model = CommittorNN_Dist_Lip(indices=np.arange(len(indices)),
                                     tri_idx1=tri_idx1,
                                     tri_idx2=tri_idx2,
                                     h1=16, h2=16, h3=8, out_dim=1, sig_k=3.0)

        rng = jax.random.PRNGKey(0)
        dummy_pos = np.zeros((1, len(indices), 3))
        params = model.init(rng, dummy_pos, training=False)
        with open(params_path, "rb") as f:
            params = serialization.from_bytes(params, f.read())
        params = jax.tree.map(
            lambda x: x.astype(np.float64) if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating) else x,
            params,
        )
        self.params = params
        self.forward_eval = make_forward_eval(model)

    @property
    def function(self):
        def wrapped_forward(pos):
            y = self.forward_eval(self.params, pos[None, :, :])
            return np.squeeze(y)
        return wrapped_forward


class Committor_CV_PIV(CollectiveVariable):
    def __init__(self, indices: List, params_path: str, blocks: Sequence):
        super().__init__(indices)

        model = CommittorNN_PIV(indices=np.arange(len(indices)), blocks=blocks, h1=32, h2=16, h3=8, out_dim=1, sig_k=3.0)

        rng = jax.random.PRNGKey(0)
        dummy_pos = np.zeros((1, len(indices), 3))
        params = model.init(rng, dummy_pos, training=False)
        with open(params_path, "rb") as f:
            params = serialization.from_bytes(params, f.read())
        params = jax.tree.map(
            lambda x: x.astype(np.float64) if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating) else x,
            params,
        )
        self.params = params
        self.forward_eval = make_forward_eval(model)

    @property
    def function(self):
        def wrapped_forward(pos):
            y = self.forward_eval(self.params, pos[None, :, :])
            return np.squeeze(y)
        return wrapped_forward

class Committor_CV_PIV_shiftsig(CollectiveVariable):
    def __init__(self, indices: List, params_path: str, blocks: Sequence, h1=32, h2=16, h3=8, sig_k=3.0):
        super().__init__(indices)

        model = CommittorNN_PIV_shiftsig(indices=np.arange(len(indices)), blocks=blocks, h1=h1, h2=h2, h3=h3, out_dim=1, sig_k=3.0)

        rng = jax.random.PRNGKey(0)
        dummy_pos = np.zeros((1, len(indices), 3))
        params = model.init(rng, dummy_pos, training=False)
        with open(params_path, "rb") as f:
            params = serialization.from_bytes(params, f.read())
        params = jax.tree.map(
            lambda x: x.astype(np.float64) if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating) else x,
            params,
        )
        self.params = params
        self.forward_eval = make_forward_eval(model)

    @property
    def function(self):
        def wrapped_forward(pos):
            y = self.forward_eval(self.params, pos[None, :, :])
            return np.squeeze(y)
        return wrapped_forward

def cartesian(idx1, idx2):
    return np.stack(np.broadcast_arrays(idx1[:, None], idx2[None, :]), axis=-1).reshape(-1, 2)
