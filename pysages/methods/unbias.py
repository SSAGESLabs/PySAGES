# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Unbiased simulation method method.

This method does not alter the original simulation.
It allows unbiased simulations with the PySAGES framework.

A common use case is to record collective variables in unbiased simulations via the Histogram logger.
"""

from typing import NamedTuple

from jax import numpy as np

from pysages.methods.core import SamplingMethod, default_getstate, generalize
from pysages.utils import JaxArray


class UnbiasState(NamedTuple):
    """
    Description of a state for unbiased simulations.

    bias: JaxArray
        Array with zero biasing force in the simulation.
    xi: JaxArray
        Collective variable value of the last simulation step.
    """

    bias: JaxArray
    xi: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class Unbias(SamplingMethod):
    """
    Unbias method class.

    Run simulations without biasing.
    """

    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, **kwargs):
        """
        Arguments
        ---------
        cvs: Union[List, Tuple]
            A list or tuple of collective variables, length `N`.
        """
        super().__init__(cvs, **kwargs)
        self.cv_dimension = len(cvs)

    def __getstate__(self):
        state, kwargs = default_getstate(self)
        return state, kwargs

    def build(self, snapshot, helpers, *args, **kwargs):
        return _unbias(self, snapshot, helpers)


def _unbias(method, snapshot, helpers):
    cv = method.cv
    natoms = np.size(snapshot.positions, 0)

    def initialize():
        bias = np.zeros((natoms, 3))
        return UnbiasState(bias, None)

    def update(state, data):
        xi, Jxi = cv(data)
        return UnbiasState(state.bias, xi)

    return snapshot, initialize, generalize(update, helpers)
