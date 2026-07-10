# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Nanoreactor simulation method method.

This method alter the original simulation by a time dependent restraint.
It allows nanoreactor simulations with the PySAGES framework.

A common use case is to sample reaction pathways.
"""

import jax.numpy as np

from pysages.methods.core import SamplingMethod, generalize
from pysages.typing import JaxArray, NamedTuple


class NanoreactorState(NamedTuple):
    """
    Description of a state for nanoreactor simulations.

    xi: JaxArray
        Collective variable value of the last simulation step.

    bias: JaxArray
        An array that stores the forces of the time dependent restraint.

    proj: JaxArray
        An array for logging extra properties.

    ncalls: int
        Counts the number of times the method's update has been called.
    """

    xi: JaxArray
    bias: JaxArray
    proj: JaxArray
    ncalls: int

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class Nanoreactor(SamplingMethod):
    """
    Nanoreactor method class.

    Run simulations with time dependent restraint.
    """

    snapshot_flags = {"positions", "indices", "masses"}

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
        self.ext_force = self.kwargs.get("ext_force", None)
        return _nanoreactor(self, snapshot, helpers)


def _nanoreactor(method, snapshot, helpers):
    cv = method.cv
    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)
    ext_force = method.ext_force

    def initialize():
        xi = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        return NanoreactorState(xi, bias, 0.0, 0)

    def update(state, data):
        ncalls = state.ncalls + 1
        xi = cv(data)
        force, proj = ext_force(data, ncalls * dt)
        bias = -force.reshape(state.bias.shape)
        return NanoreactorState(xi, bias, proj, ncalls)

    return snapshot, initialize, generalize(update, helpers)
