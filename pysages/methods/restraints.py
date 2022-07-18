# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Building blocks for specifying restraints on collective variables.
"""

from typing import NamedTuple

from jax import numpy as np

from pysages.utils import JaxArray, dispatch


class CVRestraints(NamedTuple):
    """
    Harmonic restraint parameters.

    Parameters
    ----------

    lower: JaxArray
        Collective variables lower bounds.

    upper: JaxArray
        Collective variables upper bounds.

    kl: JaxArray
        Lower restraint string constant.

    ku: JaxArray
        Upper restraint string constant.
    """

    lower: JaxArray
    upper: JaxArray
    kl: JaxArray
    ku: JaxArray


@dispatch.abstract
def canonicalize(*_):
    """
    Given `restraints`, ensures that the absolute values of the bounds are infinite when
    the corresponding restraint spring constants are zero.
    """


@dispatch
def canonicalize(restraints: CVRestraints, cvs):  # noqa: F811 # pylint: disable=C0116,E0102
    lower, upper, kl, ku = restraints

    kl = np.asarray(kl).flatten()  # flatten ensures we get vectors
    ku = np.asarray(ku).flatten()
    lower = np.where(kl == 0, -np.inf, np.asarray(lower).flatten())
    upper = np.where(ku == 0, np.inf, np.asarray(upper).flatten())

    if len(lower) != len(upper) != len(cvs):
        raise ValueError("The number of restraints must equal the number of CVs.")

    return CVRestraints(lower, upper, kl, ku)


@dispatch
def canonicalize(restraints: type(None), _):  # noqa: F811 # pylint: disable=C0116,E0102
    return restraints


def apply_restraints(lo, hi, kl, kh, xi):
    """
    Returns a harmonic force for each value in `xi` that lies outside the correspoding
    interval in `lo`..`hi`, or zero otherwise.
    """
    return np.where(xi < lo, kl * (xi - lo), np.where(xi > hi, kh * (xi - hi), 0))
