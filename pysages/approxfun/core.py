# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC
from dataclasses import dataclass
from functools import partial
from itertools import product

from jax import jit
from jax import numpy as np
from jax import vmap

from pysages.grids import Chebyshev, Grid
from pysages.typing import JaxArray, NamedTuple
from pysages.utils import dispatch


class Fun(NamedTuple):
    """
    Stores the coeffiecients of an either Fourier or Chebyshev series
    (or tensor product of series in 2D) approximating a function.
    """

    scale: JaxArray
    coefficients: JaxArray
    c0: float


@dataclass
class AbstractFit(ABC):
    """
    Stores the information necessary to approximate the coefficients
    of a basis (Fourier of Chebyshev) expansion of a function (ℝⁿ ↦ ℝ)
    that minimizes the squared error with respect to a set of given
    target values.
    """

    grid: Grid
    mesh: JaxArray
    pinv: JaxArray
    exponents: JaxArray

    def __init__(self, grid: Grid):
        ns = collect_exponents(grid)
        self.grid = grid
        self.mesh = compute_mesh(grid)
        self.exponents = ns
        self.pinv = pinv(self)

    @property
    def is_periodic(self):
        return self.grid.is_periodic


class SpectralGradientFit(AbstractFit):
    """
    Specialization of AbstractFit used when the target values correspond
    to the gradient of the function of interest.
    """

    pass


class SpectralSobolev1Fit(AbstractFit):
    """
    Specialization of AbstractFit used when the target values correspond
    to *both* the values and gradient of the function of interest.
    """

    pass


def collect_exponents(grid):
    if grid.shape.size > 2:
        raise ValueError("Only 1D and 2D grids are supported")
    #
    if grid.is_periodic:
        r = grid.shape.sum() / 4
    else:
        r = np.square(grid.shape).sum() / 16
    n = int(np.floor(np.sqrt(r))) + 1
    r = float(r)
    #
    if grid.shape.size == 1:
        return np.arange(1, n).reshape(-1, 1)
    #
    m = 1 - n if grid.is_periodic else 1
    pairs = product(range(1, n), range(m, n))
    exponents = np.vstack(
        [
            [(0, i) for i in range(1, n)],
            [(i, 0) for i in range(1, n)],
            [t for t in pairs if t[1] != 0 and t[0] ** 2 + t[1] ** 2 <= r],
        ]
    )
    return exponents


def scale(x, grid: Grid):
    """
    Applies to `x` the map that takes the `grid` to [-1, 1]ⁿ,
    where `n` is the dimensionality of `grid`.
    """
    return (x - grid.lower) * 2 / grid.size - 1


def compute_mesh(grid):
    """
    Returns a dense mesh with the same shape as `grid`, but on the hypercube [-1, 1]ⁿ,
    where `n` is the dimensionality of `grid`. The resulting mesh is Chebyshev-distributed
    if `grid: Grid[Chebyshev]`, or uniformly-distributed otherwise.
    """

    def generate_axis(n):
        transform = _generate_transform(grid, n)
        return transform(np.arange(n))

    return cartesian_product(*(generate_axis(i) for i in grid.shape))


@dispatch
def _generate_transform(_: Grid, n):
    return vmap(lambda k: -1 + (2 * k + 1) / n)


@dispatch
def _generate_transform(_: Grid[Chebyshev], n):  # noqa: F811 # pylint: disable=C0116,E0102
    return vmap(lambda k: -np.cos((k + 1 / 2) * np.pi / n))


def cartesian_product(*collections):
    """
    Given a set of `collections`, returns an array with their [Cartesian
    Product](https://en.wikipedia.org/wiki/Cartesian_product).
    """
    n = len(collections)
    coordinates = np.array(np.meshgrid(*collections, indexing="ij"))
    permutation = np.roll(np.arange(n + 1), -1)
    return np.transpose(coordinates, permutation).reshape(-1, n)


def vander_builder(grid, exponents):
    """
    Returns a closure over the grid and exponents to build a Vandermonde matrix
    of a Fourier or Chebyshev basis expansion.
    """
    ns = exponents

    if grid.is_periodic:

        def expand(x):
            return np.exp(1j * np.pi * ns * x).prod(axis=1).T

    else:

        def expand(x):
            return (x**ns).prod(axis=1).T

    return jit(lambda xs: vmap(expand)(xs).reshape(-1, np.size(ns, 0)))


def vandergrad_builder(grid, exponents):
    """
    Returns a closure over the grid and exponents to build a Vandermonde-like
    matrix for fitting the gradient of a Fourier or Chebyshev expansion.
    """
    s = 2 * (np.pi if grid.is_periodic else 1) / grid.size
    ns = exponents

    if grid.shape.size == 1:

        def flip_multiply(x, y):
            return x

    else:

        def flip_multiply(x, y):
            return x * np.fliplr(y)

    if grid.is_periodic:

        def expand(x):
            z = np.exp(-1j * np.pi * ns * x)
            return flip_multiply(s * ns * z, z).T

    else:

        def expand(x):
            z = x ** (np.maximum(ns - 1, 0))
            return flip_multiply(s * ns * z, x**ns).T

    return jit(lambda xs: vmap(expand)(xs).reshape(-1, np.size(ns, 0)))


@dispatch
def pinv(model: SpectralGradientFit):
    A = vandergrad_builder(model.grid, model.exponents)(model.mesh)

    if model.is_periodic:
        A = np.hstack((np.imag(A), np.real(A)))

    return np.linalg.pinv(A)


@dispatch
def pinv(model: SpectralSobolev1Fit):  # noqa: F811 # pylint: disable=C0116,E0102
    ns = model.exponents
    A = vander_builder(model.grid, ns)(model.mesh)
    B = vandergrad_builder(model.grid, ns)(model.mesh)
    Y = np.ones((np.size(A, 0), 1))
    Z = np.zeros((np.size(B, 0), 1))

    if model.is_periodic:
        U = np.hstack((Y, np.real(A), np.imag(A)))
        V = np.hstack((Z, np.imag(B), np.real(B)))
    else:
        U = np.hstack((Y, A))
        V = np.hstack((Z, B))

    return np.linalg.pinv(np.vstack((U, V)))


@dispatch
def build_fitter(model: SpectralGradientFit):
    """
    Returns a function which takes an approximation `dy` to the gradient
    of a function `f` evaluated over the set `x = compute_mesh(model.grid)`,
    and returns a `fun: Fun` object which approximates `f` over
    the domain [-1, 1]ⁿ.
    """
    axes = tuple(range(model.grid.shape.size))

    if model.is_periodic:

        def fit(dy):
            std = dy.std(axis=axes).max()
            std = np.where(std == 0, 1, std)
            dy = (dy - dy.mean(axis=axes)) / std
            return Fun(std, model.pinv @ dy.flatten(), np.array(0.0))

    else:

        def fit(dy):
            std = dy.std(axis=axes).max()
            std = np.where(std == 0, 1, std)
            dy = dy / std
            return Fun(std, model.pinv @ dy.flatten(), np.array(0.0))

    return jit(fit)


@dispatch
def build_fitter(model: SpectralSobolev1Fit):  # noqa: F811 # pylint: disable=C0116,E0102
    """
    Returns a function which takes approximations `y` and dy` to a function
    `f` and its gradient evaluated over the set `x = compute_mesh(model.grid)`,
    and in turn returns a `fun: Fun` object which approximates `f` over
    the domain [-1, 1]ⁿ.
    """
    axes = tuple(range(model.grid.shape.size))

    def fit(y, dy):
        mean = y.mean()
        std = np.maximum(y.std(), dy.std(axis=axes).max())
        std = np.where(std == 0, 1, std)
        y = (y - mean) / std
        dy = dy / std
        cs = model.pinv @ np.hstack((y.flatten(), dy.flatten()))
        return Fun(std, cs[1:], cs[0] + mean / std)

    return jit(fit)


def build_evaluator(model):
    """
    Returns a method to evaluate the Fourier or Chebyshev expansion defined
    by `model`. The returned method takes a `fun` and a value (or array of
    values) `x` to evaluate the approximant.
    """
    transform = partial(scale, grid=model.grid)
    vander = vander_builder(model.grid, model.exponents)

    if model.is_periodic:

        def restack(x):
            return np.hstack((np.real(x), np.imag(x)))

    else:

        def restack(x):
            return x

    def evaluate(f: Fun, x):
        c0 = f.c0
        cs = f.coefficients
        x = transform(np.array(x, ndmin=2))
        y = f.scale * (restack(vander(x)) @ cs + c0)
        return y.reshape(np.size(x, 0), -1)

    return jit(evaluate)


def build_grad_evaluator(model):
    """
    Returns a method to evaluate the gradient of a Fourier or Chebyshev
    expansion defined by `model`. The returned method takes a `fun` and a value
    (or array of values) `x` to evaluate the approximant.
    """
    transform = partial(scale, grid=model.grid)
    vandergrad = vandergrad_builder(model.grid, model.exponents)

    if model.is_periodic:

        def restack(x):
            return np.hstack((np.imag(x), np.real(x)))

    else:

        def restack(x):
            return x

    def get_gradient(f: Fun, x):
        cs = f.coefficients
        x = transform(np.array(x, ndmin=2))
        y = f.scale * (restack(vandergrad(x)) @ cs)
        return y.reshape(x.shape)

    return jit(get_gradient)
