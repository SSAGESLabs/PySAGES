# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import jax.numpy as np
from jax.numpy import linalg
from .core import TwoPointCV, AxisCV


def barycenter(positions):
    return np.sum(positions, axis=0) / positions.shape[0]


def weighted_barycenter(positions, weights):
    n = positions.shape[0]
    R = np.zeros(3)
    # TODO: Replace by `np.sum` and `vmap`
    for i in range(n):
        w, r = weights[i], positions[i]
        R += w * r
    return R


class Component(AxisCV):
    @property
    def function(self):
        return (lambda rs: barycenter(rs)[self.axis])


class Distance(TwoPointCV):
    @property
    def function(self):
        return distance


def distance(r1, r2):
    return linalg.norm(r1 - r2)
