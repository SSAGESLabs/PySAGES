# from matplotlib import pyplot as plt
from jax import grad, numpy as np, vmap

from pysages.grids import Chebyshev, Grid
from pysages.approxfun import (
    SpectralGradientFit,
    SpectralSobolev1Fit,
    build_fitter,
    build_evaluator,
    build_grad_evaluator,
    compute_mesh,
)


# Test functions
def gaussian(a, mu, sigma, x):
    return a * np.exp(-((x - mu) ** 2) / sigma)


def g(x):
    return gaussian(1.5, 0.2, 0.03, x) + gaussian(0.5, -0.5, 0.05, x) + gaussian(1.25, 0.9, 1.5, x)


def f(x):
    return 0.35 * np.cos(5 * x) + 0.7 * np.sin(-2 * x)


def test_fourier_approx():
    grid = Grid(lower=(-np.pi,), upper=(np.pi,), shape=(512,), periodic=True)

    x = np.pi * compute_mesh(grid)

    # Periodic function and its gradient
    y = vmap(f)(x.flatten()).reshape(x.shape)
    dy = vmap(grad(f))(x.flatten()).reshape(x.shape)

    model = SpectralGradientFit(grid)
    fit = build_fitter(model)
    evaluate = build_evaluator(model)
    get_grad = build_grad_evaluator(model)

    fun = fit(dy)

    assert np.all(np.isclose(y, evaluate(fun, x))).item()
    assert np.all(np.isclose(dy, get_grad(fun, x))).item()

    # fig, ax = plt.subplots()
    # ax.plot(x, dy)
    # ax.plot(x, get_grad(fun, x))
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.plot(x, evaluate(fun, x))
    # plt.show()

    model = SpectralSobolev1Fit(grid)
    fit = build_fitter(model)
    evaluate = build_evaluator(model)
    get_grad = build_grad_evaluator(model)

    sfun = fit(y, dy)

    assert np.all(np.isclose(y, evaluate(sfun, x))).item()
    assert np.all(np.isclose(dy, get_grad(sfun, x))).item()

    assert np.linalg.norm(fun.coefficients - sfun.coefficients) < 1e-8

    # fig, ax = plt.subplots()
    # ax.plot(x, dy)
    # ax.plot(x, get_grad(sfun, x))
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.plot(x, evaluate(sfun, x))
    # plt.show()


def test_cheb_approx():
    grid = Grid[Chebyshev](lower=(-1.0,), upper=(1.0,), shape=(256,))

    x = compute_mesh(grid)

    y = vmap(g)(x.flatten()).reshape(x.shape)
    dy = vmap(grad(g))(x.flatten()).reshape(x.shape)

    model = SpectralSobolev1Fit(grid)
    fit = build_fitter(model)
    evaluate = build_evaluator(model)
    get_grad = build_grad_evaluator(model)

    fun = fit(y, dy)

    assert np.linalg.norm(y - evaluate(fun, x)) < 1e-2
    assert np.linalg.norm(dy - get_grad(fun, x)) < 5e-2

    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.plot(x, evaluate(fun, x))
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(x, dy)
    # ax.plot(x, get_grad(fun, x))
    # plt.show()
