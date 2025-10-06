from functools import partial

import matplotlib.pyplot as plt
from jax import grad, jit
from jax import numpy as np
from jax import vmap

from pysages.approxfun import compute_mesh
from pysages.approxfun import scale as _scale
from pysages.approxfun import unit_mesh
from pysages.grids import Chebyshev, Grid
from pysages.ml.models import MLP, Siren
from pysages.ml.objectives import L2Regularization, Sobolev1SSE
from pysages.ml.optimizers import JaxOptimizer, LevenbergMarquardt
from pysages.ml.training import build_fitting_function
from pysages.ml.utils import pack, unpack
from pysages.utils import try_import

jopt = try_import("jax.example_libraries.optimizers", "jax.experimental.optimizers")


# Test functions
def gaussian(a, mu, sigma, x):
    return a * np.exp(-((x - mu) ** 2) / sigma)


def g(x):
    return gaussian(1.5, 0.2, 0.03, x) + gaussian(0.5, -0.5, 0.05, x) + gaussian(1.25, 0.9, 1.5, x)


def f(x):
    return 0.35 * np.cos(5 * x) + 0.7 * np.sin(-2 * x)


def nngrad(model, params):
    d = grad(lambda x: model.apply(params, x.reshape(-1, 1)).sum())
    return jit(vmap(d))


def test_siren_sobolev_training():
    grid = Grid(lower=(-np.pi,), upper=(np.pi,), shape=(64,), periodic=True)
    scale = partial(_scale, grid=grid)

    x = compute_mesh(grid)

    # Periodic function and its gradient
    y = vmap(f)(x.flatten()).reshape(x.shape)
    dy = vmap(grad(f))(x.flatten()).reshape(x.shape)

    topology = (4, 4)
    model = Siren(1, 1, topology, transform=scale)
    optimizer = LevenbergMarquardt(loss=Sobolev1SSE(), max_iters=200)
    fit = build_fitting_function(model, optimizer)

    ps, layout = unpack(model.parameters)
    params = fit(ps, x, (y, dy)).params
    params = jit(lambda ps: pack(ps, layout))(params)

    assert np.linalg.norm(y - model.apply(params, x)).item() / x.size < 5e-5
    assert np.linalg.norm(dy - nngrad(model, params)(x)).item() / x.size < 5e-4

    x_plot = np.linspace(-np.pi, np.pi, 512)

    fig, ax = plt.subplots()
    ax.plot(x_plot, vmap(f)(x_plot))
    ax.plot(x_plot, model.apply(params, x_plot), linestyle="dashed")
    fig.savefig("y_periodic_sirens_sobolev_fit.pdf")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x_plot, vmap(grad(f))(x_plot))
    ax.plot(x_plot, nngrad(model, params)(x_plot), linestyle="dashed")
    fig.savefig("dy_periodic_sirens_sobolev_fit.pdf")
    plt.close(fig)


def test_mlp_training():
    grid = Grid[Chebyshev](lower=(-1.0,), upper=(1.0,), shape=(64,))

    x = unit_mesh(grid)

    y = vmap(g)(x.flatten()).reshape(x.shape)

    topology = (4, 4)
    model = MLP(1, 1, topology)
    optimizer = LevenbergMarquardt(reg=L2Regularization(0.0))
    fit = build_fitting_function(model, optimizer)

    params, layout = unpack(model.parameters)
    params = fit(params, x, y).params
    y_model = model.apply(pack(params, layout), x)

    assert np.linalg.norm(y - y_model).item() / x.size < 5e-4

    x_plot = np.linspace(-1, 1, 512)
    fig, ax = plt.subplots()
    ax.plot(x_plot, vmap(g)(x_plot))
    ax.plot(x_plot, model.apply(pack(params, layout), x_plot), linestyle="dashed")
    fig.savefig("y_mlp_fit.pdf")
    plt.close(fig)


def test_adam_optimizer():
    grid = Grid[Chebyshev](lower=(-1.0,), upper=(1.0,), shape=(128,))

    x = unit_mesh(grid)

    y = vmap(g)(x.flatten()).reshape(x.shape)

    topology = (4, 4)
    model = MLP(1, 1, topology)
    optimizer = JaxOptimizer(jopt.adam, tol=1e-6)
    fit = build_fitting_function(model, optimizer)

    params, layout = unpack(model.parameters)
    params = fit(params, x, y).params
    y_model = model.apply(pack(params, layout), x)

    assert np.linalg.norm(y - y_model).item() / x.size < 1e-2

    x_plot = np.linspace(-1, 1, 512)
    fig, ax = plt.subplots()
    ax.plot(x_plot, vmap(g)(x_plot))
    ax.plot(x_plot, model.apply(pack(params, layout), x_plot), linestyle="dashed")
    fig.savefig("y_mlp_adam_fit.pdf")
    plt.close(fig)
