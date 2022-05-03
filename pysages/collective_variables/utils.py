# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Extra utilities for computations involving collective variables.
"""

from jax import numpy as np

from pysages.collective_variables.angles import Angle, DihedralAngle


def get_periods(cvs):
    """
    Returns an array with `2 * np.pi` for each cv in `cvs` that is of periodic type
    and `inf` for the rest of the entries.
    """
    periodic_types = (Angle, DihedralAngle)
    return np.array([2 * np.pi if type(cv) in periodic_types else np.inf for cv in cvs])


def wrap(x, P):
    """
    Given a period `P`, wraps around `x` over the interval from `-P / 2` to `P / 2`.
    """
    return np.where(np.isinf(P), x, x - (np.abs(x) > P / 2) * np.sign(x) * P)
