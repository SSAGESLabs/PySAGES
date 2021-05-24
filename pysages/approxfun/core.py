# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from collections import namedtuple
from itertools import product
from jax import jit, vmap

import jax.numpy as np


class Fun(namedtuple(
    "Fun",
    (
        "coefficients",
        "c0",
    )
)):
    pass


class VandermondeGradientFit(namedtuple(
    "VandermondeGradientFit",
    (
        "pinvA",
        "exponents",
        "is_periodic",
    )
)):
    def __new__(cls, grid):
        mesh = compute_mesh(grid)
        ns = collect_exponents(grid)
        pinvA = vanderpinv(grid, ns, mesh)
        return super().__new__(cls, pinvA, ns, grid.is_periodic)


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
    exponents = np.vstack([
        [(0, i) for i in range(1, n)],
        [(i, 0) for i in range(1, n)],
        [t for t in pairs if t[1] != 0 and t[0]**2 + t[1]**2 <= r]
    ])
    return exponents


def compute_mesh(grid):
    if grid.is_periodic:
        h = grid.size / grid.shape
        o = -np.pi + h / 2
        nodes = o + h * np.hstack(
            [np.arange(i).reshape(-1, 1) for i in grid.shape]
        )
    else:
        def transform(n):
            return vmap(lambda k: -np.cos((k + 1 / 2) * np.pi / n))
        #
        nodes = np.hstack(
            [transform(i)(np.arange(i).reshape(-1, 1)) for i in grid.shape]
        )
    components = np.meshgrid(*nodes.T,)
    return np.hstack([v.reshape(-1, 1) for v in components])


def build_mapping(grid, exponents):
    ns = exponents
    #
    if grid.shape.size == 1:
        def flip_multiply(x, y):
            return x
    else:
        def flip_multiply(x, y):
            return x * np.fliplr(y)
    #
    if grid.is_periodic:
        def expand(x):
            z = np.exp(1j * ns * x)
            return flip_multiply(-ns * z, z).T
    else:
        def expand(x):
            z = x**(np.maximum(ns - 1, 0))
            return flip_multiply(ns * z, x).T
    #
    return jit(lambda xs: vmap(expand)(xs).reshape(-1, np.size(ns, 0)))


def vanderpinv(grid, exponents, mesh):
    ns = exponents
    expand = build_mapping(grid, ns)
    A = expand(mesh)
    #
    if grid.is_periodic:
        A = np.hstack((np.imag(A), np.real(A)))
    #
    return np.linalg.pinv(A)


def build_fitter(model: VandermondeGradientFit):

    if model.is_periodic:
        def fit(data):
            μ = data.mean(axis = 0)
            dA = data - μ
            return Fun(model.pinvA @ dA.flatten(), μ)
    else:
        def fit(data):
            return Fun(model.pinvA @ data.flatten(), np.array(0.0))
    #
    return jit(fit)


def build_interpolator(model: VandermondeGradientFit, grid):
    ns = model.exponents
    expand = build_mapping(grid, ns)
    #
    if grid.is_periodic:
        def restack(x):
            return np.hstack((np.imag(x), np.real(x)))
    else:
        def restack(x):
            return x

    def interpolate(f: Fun, x):
        c0 = f.c0
        cs = f.coefficients
        return restack(expand(x)) @ cs + c0
    #
    return jit(interpolate)


def build_integrator(model: VandermondeGradientFit, grid):
    ns = model.exponents
    #
    if grid.is_periodic:
        def _expand(x):
            return np.prod(np.exp(1j * ns * x), axis = 1).T

        def restack(x):
            return np.hstack((-np.real(x), np.imag(x)))
    else:
        def _expand(x):
            return np.prod(x**ns, axis = 1).T

        def restack(x):
            return x
    #
    expand = jit(lambda x: vmap(_expand)(x).reshape(-1, np.size(ns, 0)))

    def integrate(f: Fun, x):
        cs = f.coefficients
        return restack(expand(x)) @ cs
    #
    return jit(integrate)
