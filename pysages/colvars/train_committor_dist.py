#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX/Flax implementation of your committor NN with Lipschitz-normalized
linear layers, boundary + gradient + Lipschitz losses, and an Optax
training loop. This version fixes the earlier JIT issue by closing over
`loss_fn` instead of passing it as an argument to a jitted function.

Usage (example):
  # (optional) to avoid accidental GPU probing / driver issues
  # export JAX_PLATFORMS=cpu
  python run_committor_flax.py

Notes:
- Keep this file name NOT equal to 'jax.py' to avoid shadowing the JAX lib.
- Labels: 0=A (reactant), 1=sim/intermediate, 2=B (product).
- Saves params to distance_flax.params and a histogram to fig_jax.png.
"""

import os
# Respect external setting; uncomment the next line to force CPU:
# os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")  # float32 for speed

from typing import Any, Dict, Tuple, Iterable, Sequence
import numpy as onp
import jax
import jax.numpy as np
from flax import linen as nn
from flax.training import train_state
import optax

# ------------------------------
# Layers
# ------------------------------
class NormalizedLinear(nn.Module):
    in_features: int
    out_features: int

    @nn.compact
    def __call__(self, x: np.ndarray) -> np.ndarray:
        kernel = self.param(
            "kernel", nn.initializers.lecun_normal(), (self.out_features, self.in_features)
        )
        bias = self.param("bias", nn.initializers.zeros, (self.out_features,))
        ci = self.param("ci", nn.initializers.ones, ())  # trainable scalar

        absrowsum = np.sum(np.abs(kernel), axis=1, keepdims=True)
        scale = np.minimum(1.0, nn.softplus(ci) / (absrowsum + 1e-12))
        w_norm = kernel * scale
        return np.dot(x, w_norm.T) + bias


class CommittorNN_Dist_Lip(nn.Module):
    indices: np.ndarray   # (M,)
    tri_idx1: np.ndarray  # (K,)
    tri_idx2: np.ndarray  # (K,)
    h1: int = 16
    h2: int = 16
    h3: int = 8
    out_dim: int = 1
    sig_k: float = 3.0

    @nn.compact
    def __call__(self, pos: np.ndarray, training: bool = False) -> np.ndarray:
        # pos: (B, N, 3)
        r = pos[:, self.indices, :]                                 # (B, M, 3)
        diffs = r[:, self.tri_idx1, :] - r[:, self.tri_idx2, :]     # (B, K, 3)
        dists = np.linalg.norm(diffs, ord=2, axis=-1)               # (B, K)

        x = dists
        x = NormalizedLinear(dists.shape[-1], self.h1)(x); x = np.tanh(x)
        x = NormalizedLinear(self.h1, self.h2)(x);       x = np.tanh(x)
        x = NormalizedLinear(self.h2, self.h3)(x);       x = np.tanh(x)
        x = NormalizedLinear(self.h3, self.out_dim)(x)              # (B,1)
        x = np.squeeze(x, axis=-1)                                  # (B,)
        return jax.nn.sigmoid(self.sig_k * x) if training else x


def get_block_dist(pos, block):
    return np.sort(np.linalg.norm(pos[:,block[:,0]] - pos[:,block[:,1]], axis=-1), axis=-1)

def get_x(pos, blocks):
    return np.concatenate([get_block_dist(pos, block) for block in blocks], axis=-1)

class CommittorNN_PIV(nn.Module):
    indices: np.ndarray   # (M,)
    blocks: Sequence
    h1: int = 32
    h2: int = 16
    h3: int = 8
    out_dim: int = 1
    sig_k: float = 3.0

    @nn.compact
    def __call__(self, pos: np.ndarray, training: bool = False) -> np.ndarray:
        # pos: (B, N, 3)
        x = get_x(pos[:, self.indices, :], self.blocks)
        x = NormalizedLinear(x.shape[-1], self.h1)(x); x = np.tanh(x)
        x = NormalizedLinear(self.h1, self.h2)(x);	 x = np.tanh(x)
        x = NormalizedLinear(self.h2, self.h3)(x);	 x = np.tanh(x)
        x = NormalizedLinear(self.h3, self.out_dim)(x)              # (B,1)
        x = np.squeeze(x, axis=-1)                                  # (B,)
        return jax.nn.sigmoid(self.sig_k * x) if training else x

# ------------------------------
# Losses
# ------------------------------
@jax.jit
def lipschitz_loss_from_params(params: Dict[str, Any]) -> np.ndarray:
    prod = 1.0

    def walk(p):
        nonlocal prod
        if isinstance(p, dict):
            if "ci" in p:
                prod = prod * nn.softplus(p["ci"])  # multiply scalar
            for v in p.values():
                walk(v)

    walk(params)
    return prod


def make_loss_fn(model: nn.Module,
                 masses: np.ndarray,
                 boundary_weight: float = 100.0,
                 lipschitz_weight: float = 3e-3,
                 gradient_weight: float = 1.0):
    # masses: (1, N, 1) broadcast to (B, N, 1)
    def loss_fn(params, pos_batch: np.ndarray, labels: np.ndarray, weights: np.ndarray):
        q = model.apply(params, pos_batch, training=True)  # (B,)

        # boundary term (average over non-1 labels)
        is_A = (labels == 0)
        is_B = (labels == 2)
        is_1 = (labels == 1)
        num_1 = np.maximum(1, np.sum(is_1))
        num_not1 = np.maximum(1, pos_batch.shape[0] - np.sum(is_1))
        boundary = (np.sum((q**2) * is_A) + np.sum(((q - 1.0)**2) * is_B)) / num_not1

        # gradient term (only on label==1)
        def q_sum_over_batch_pos(pos):
            return np.sum(model.apply(params, pos, training=True))
        grad_pos = jax.grad(q_sum_over_batch_pos)(pos_batch)  # (B,N,3)
        grad_sq = (grad_pos**2) / masses
        grad_per_sample = np.sum(grad_sq, axis=(1, 2))  # (B,)
        grad_loss = np.sum(np.where(is_1, weights * grad_per_sample, 0.0)) / num_1

        # lipschitz product of softplus(ci)
        lip = lipschitz_loss_from_params(params)

        total = 1e4 * (gradient_weight * grad_loss + boundary_weight * boundary + lipschitz_weight * lip)
        return total, (1e4 * gradient_weight * grad_loss,
                       1e4 * boundary_weight * boundary,
                       1e4 * lipschitz_weight * lip)

    return loss_fn


# ------------------------------
# Train state & steps
# ------------------------------
class TrainState(train_state.TrainState):
    pass


def create_train_state(rng, model, learning_rate: float, pos_shape: Tuple[int, int, int]):
    params = model.init(rng, np.zeros(pos_shape, onp.float32), training=True)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=1e-5),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step(loss_fn):
    @jax.jit
    def train_step(state: TrainState, pos_b, labels_b, weights_b):
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, pos_b, labels_b, weights_b
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, parts
    return train_step


def make_eval_step(loss_fn):
    @jax.jit
    def eval_step(params, pos_b, labels_b, weights_b):
        loss, parts = loss_fn(params, pos_b, labels_b, weights_b)
        return loss, parts
    return eval_step


# ------------------------------
# Data utils
# ------------------------------
def make_batches(pos: onp.ndarray, labels: onp.ndarray, weights: onp.ndarray,
                 batch_size: int, shuffle: bool = True,
                 rng: onp.random.Generator | None = None) -> Iterable[Tuple[onp.ndarray, onp.ndarray, onp.ndarray]]:
    N = pos.shape[0]
    idx = onp.arange(N)
    if shuffle:
        (rng or onp.random.default_rng()).shuffle(idx)
    n_full = N // batch_size  # drop_last
    idx = idx[: n_full * batch_size].reshape(n_full, batch_size)
    for row in idx:
        yield pos[row], labels[row], weights[row]


# ------------------------------
# Training loop
# ------------------------------

def train(model: nn.Module,
          masses: onp.ndarray,
          train_data: Tuple[onp.ndarray, onp.ndarray, onp.ndarray],
          val_data: Tuple[onp.ndarray, onp.ndarray, onp.ndarray],
          batch_size: int = 1024,
          num_epochs: int = 100,
          lr: float = 1e-3,
          seed: int | None = 61982):
    rng = jax.random.PRNGKey(seed if seed is not None else 0)
    pos_train, y_train, w_train = train_data
    pos_val, y_val, w_val = val_data

    _, N, D = pos_train.shape
    state = create_train_state(rng, model, lr, (batch_size, N, D))

    loss_fn = make_loss_fn(model, np.asarray(masses),
                           boundary_weight=100.0,
                           lipschitz_weight=3e-3,
                           gradient_weight=1.0)
    train_step = make_train_step(loss_fn)
    eval_step = make_eval_step(loss_fn)

    for epoch in range(num_epochs):
        tr_losses = []; tr_grad = []; tr_bound = []; tr_lip = []
        for xb, yb, wb in make_batches(pos_train, y_train, w_train, batch_size, shuffle=True):
            xb = np.asarray(xb, dtype=onp.float32)
            yb = np.asarray(yb, dtype=onp.int32)
            wb = np.asarray(wb, dtype=onp.float32)
            state, loss, parts = train_step(state, xb, yb, wb)
            tr_losses.append(float(loss)); g,b,l = parts; tr_grad.append(float(g)); tr_bound.append(float(b)); tr_lip.append(float(l))

        va_losses = []; va_grad = []; va_bound = []; va_lip = []
        for xb, yb, wb in make_batches(pos_val, y_val, w_val, batch_size, shuffle=False):
            xb = np.asarray(xb, dtype=onp.float32)
            yb = np.asarray(yb, dtype=onp.int32)
            wb = np.asarray(wb, dtype=onp.float32)
            loss, parts = eval_step(state.params, xb, yb, wb)
            va_losses.append(float(loss)); g,b,l = parts; va_grad.append(float(g)); va_bound.append(float(b)); va_lip.append(float(l))

        print(f"Epoch {epoch}")
        if tr_losses:
            print("  avg train loss:", onp.mean(tr_losses))
            print("  avg train grad loss:", onp.mean(tr_grad))
            print("  avg train bound loss:", onp.mean(tr_bound))
            print("  avg train lipschitz loss:", onp.mean(tr_lip))
        if va_losses:
            print("  avg val loss:", onp.mean(va_losses))
            print("  avg val grad loss:", onp.mean(va_grad))
            print("  avg val bound loss:", onp.mean(va_bound))
            print("  avg val lipschitz loss:", onp.mean(va_lip))
        print()

    return state


# ------------------------------
# Inference helper
# ------------------------------
def make_forward_eval(model):
    @jax.jit
    def _forward_eval(params, pos_b):
        return model.apply(params, pos_b, training=False)
    return _forward_eval


# ------------------------------
# Main: mirrors your IO (MDAnalysis/pysages)
# ------------------------------
if __name__ == "__main__":
    import MDAnalysis as mda
    import pysages
    from ase.io import read as ase_read
    import matplotlib.pyplot as plt

    # ----- Load CV and weights -----
    xi = onp.loadtxt("/scratch/aq2212/CV.log")[:, 1][::10]
    run_result = pysages.load("/scratch/aq2212/production_abf2_equil15/restart.pkl")
    fe_result = pysages.analyze(run_result)
    biases = fe_result['fes_fn'](xi)
    biases /= (8.6173e-5 * 600)
    biases = biases[:, 0]
    weightsSim = onp.exp(-biases).astype(onp.float32)

    # ----- Extract downsampled sim frames -----
    u = mda.Universe("sampled.xyz")
    posSim = [u.atoms.positions.copy() for _ in u.trajectory]
    posSim = onp.asarray(posSim, dtype=onp.float32)

    # ----- Load endpoints -----
    def load_xyz(path):
        uu = mda.Universe(path)
        return onp.asarray([uu.atoms.positions.copy() for _ in uu.trajectory], dtype=onp.float32)

    posA = load_xyz("/scratch/projects/depablolab/acqin2/lipschitz/visnet/model_for_sim1/A.xyz")
    posB = load_xyz("/scratch/projects/depablolab/acqin2/lipschitz/visnet/model_for_sim1/B.xyz")
    posC = load_xyz("/scratch/projects/depablolab/acqin2/lipschitz/visnet/model_for_sim1/C.xyz")

    labelsA = onp.zeros((len(posA),), dtype=onp.int32)
    labelsB = onp.full((len(posB),), 2, dtype=onp.int32)
    labelsC = onp.full((len(posC),), 1, dtype=onp.int32)
    labelsSim = onp.full((len(posSim),), 1, dtype=onp.int32)

    weightsA = onp.ones((len(posA),), dtype=onp.float32)
    weightsB = onp.ones((len(posB),), dtype=onp.float32)
    weightsC = onp.full((len(posC),), 1e-3, dtype=onp.float32)

    # masses shape (1, N, 1)
    atoms0 = ase_read('/scratch/projects/depablolab/acqin2/lipschitz/visnet/model_for_sim1/A.xyz', index=0)
    masses = onp.asarray(atoms0.get_masses(), dtype=onp.float32)[None, :, None]

    # ----- Train/val split (A+B+Sim)
    pos_all = onp.concatenate([posA, posB, posSim], axis=0)
    lab_all = onp.concatenate([labelsA, labelsB, labelsSim], axis=0)
    w_all = onp.concatenate([weightsA, weightsB, weightsSim], axis=0)

    rng_np = onp.random.default_rng(61982)
    idx = onp.arange(len(pos_all)); rng_np.shuffle(idx)
    split = int(0.7 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    train_data = (pos_all[tr_idx], lab_all[tr_idx], w_all[tr_idx])
    val_data = (pos_all[va_idx], lab_all[va_idx], w_all[va_idx])

    # ----- Model config (indices / triplets)
    indices = np.arange(50)
    tri_idx1 = np.asarray([44, 47, 47, 6, 5, 47, 7, 5])
    tri_idx2 = np.asarray([46, 46, 49, 49, 44, 49, 46, 7])

    model = CommittorNN_Dist_Lip(indices=indices,
                                 tri_idx1=tri_idx1,
                                 tri_idx2=tri_idx2,
                                 h1=16, h2=16, h3=8, out_dim=1, sig_k=3.0)

    state = train(model,
                  masses=masses,
                  train_data=train_data,
                  val_data=val_data,
                  batch_size=10000,
                  num_epochs=200,  # consider reducing for quick tests
                  lr=1e-3,
                  seed=61982)

    # ----- Save params -----
    import flax.serialization as serialization
    param_bytes = serialization.to_bytes(state.params)
    with open("distance_flax.params", "wb") as f:
        f.write(param_bytes)

    forward_eval = make_forward_eval(model)

    # ----- Evaluate on A/B/C and plot -----
    def batched_preds(pos_arr, bs=2048):
        outs = []
        for xb, _, _ in make_batches(pos_arr,
                                     onp.zeros(len(pos_arr), onp.int32),
                                     onp.ones(len(pos_arr),  onp.float32),
                                     bs, shuffle=False):
            xb = np.asarray(xb, dtype=onp.float32)
            y  = forward_eval(state.params, xb)
            outs.append(onp.asarray(y))
        return onp.concatenate(outs, axis=0) if outs else onp.zeros((0,), dtype=onp.float32)

    As = batched_preds(posA)
    Bs = batched_preds(posB)
    Cs = batched_preds(posC)

    import matplotlib
    plt.figure()
    plt.hist([As, Bs, Cs], bins=100, label=['Reactant', 'Product', 'Intermediate'],
             alpha=0.3, histtype='stepfilled', density=True)
    plt.legend(); plt.xlabel('Output value'); plt.ylabel('Probability Density')
    plt.title('Histogram of outputs by class')
    plt.savefig('fig_jax.png', dpi=200)

    print("Saved: distance_flax.params, fig_jax.png")
