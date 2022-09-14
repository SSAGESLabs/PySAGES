# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Unbiased simulation method method.

This method does not alter the original simulation.
It allows unbiased simulations with the PySAGES framework.

A common use case is to record collective variables in unbiased
simulations via the Histogram logger.
"""

from typing import NamedTuple

from pysages.methods.core import SamplingMethod, generalize
from pysages.utils import JaxArray


class UnbiasedState(NamedTuple):
    """
    Description of a state for unbiased simulations.

    xi: JaxArray
        Collective variable value of the last simulation step.

    bias: JaxArray
        Array with zero biasing force in the simulation.
    """

    xi: JaxArray
    bias: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class Unbiased(SamplingMethod):
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
        kwargs["cv_grad"] = False
        super().__init__(cvs, **kwargs)

    def build(self, snapshot, helpers, *args, **kwargs):
        return _unbias(self, snapshot, helpers)


def _unbias(method, snapshot, helpers):
    cv = method.cv

    def initialize():
        xi = cv(helpers.query(snapshot))
        return UnbiasedState(xi, None)

    def update(state, data):
        xi = cv(data)
        return UnbiasedState(xi, None)

    return snapshot, initialize, generalize(update, helpers)
