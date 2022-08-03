# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Harmonic bias method.

Biasing a simulation towards a value of a collective variable is the foundation of a
number of advanced sampling methods - umbrella integration, WHAM, string method to name a few.
This method implements such a bias.

The Hamiltonian is amended with a term
:math:`\\mathcal{H} = \\mathcal{H}_0 + \\mathcal{H}_\\mathrm{HB}(\\xi)` where
:math:`\\mathcal{H}_\\mathrm{HB}(\\xi) = \\boldsymbol{K}/2 (\\xi_0 - \\xi)^2`
biases the simulations around the collective variable :math:`\\xi_0`.
"""

from typing import NamedTuple

from jax import numpy as np

from pysages.methods.core import generalize
from pysages.methods.bias import Bias
from pysages.utils import JaxArray


class HarmonicBiasState(NamedTuple):
    """
    Description of a state biased by a harmonic potential for a CV.

    bias: JaxArray
        Array with harmonic biasing forces for each particle in the simulation.
    xi: JaxArray
        Collective variable value of the last simulation step.
    """

    bias: JaxArray
    xi: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class HarmonicBias(Bias):
    """
    Harmonic bias method class.
    """

    __special_args__ = Bias.__special_args__.union({"kspring"})

    def __init__(self, cvs, kspring, center, **kwargs):
        """
        Arguments
        ---------
        cvs: Union[List, Tuple]
            A list or tuple of collective variables, length `N`.
        kspring:
            A scalar, array length `N` or symmetric `N x N` matrix. Restraining spring constant.
        center:
            An array of length `N` representing the minimum of the harmonic biasing potential.
        """
        super().__init__(cvs, center, **kwargs)
        self.cv_dimension = len(cvs)
        self.kspring = kspring

    def __getstate__(self):
        state, kwargs = super().__getstate__()
        state["kspring"] = self._kspring
        return state, kwargs

    @property
    def kspring(self):
        """
        Retrieve the spring constant.
        """
        return self._kspring

    @kspring.setter
    def kspring(self, kspring):
        """
        Set new spring constant.

        Arguments
        ---------
        kspring:
            A scalar, array length `N` or symmetric `N x N` matrix. Restraining spring constant.
        """
        # Ensure array
        kspring = np.asarray(kspring)
        shape = kspring.shape
        N = self.cv_dimension

        if len(shape) > 2:
            raise RuntimeError(f"Wrong kspring shape {shape} (expected scalar, 1D or 2D)")
        if len(shape) == 2:
            if shape != (N, N):
                raise RuntimeError(f"2D kspring with wrong shape, expected ({N}, {N}), got {shape}")
            if not np.allclose(kspring, kspring.T):
                raise RuntimeError("Spring matrix is not symmetric")

            self._kspring = kspring
        else:  # len(shape) == 0 or len(shape) == 1
            kspring_size = kspring.size
            if kspring_size not in (N, 1):
                raise RuntimeError(f"Wrong kspring size, expected 1 or {N}, got {kspring_size}.")

            self._kspring = np.identity(N) * kspring
        return self._kspring

    def build(self, snapshot, helpers, *args, **kwargs):
        return _harmonic_bias(self, snapshot, helpers)


def _harmonic_bias(method, snapshot, helpers):
    cv = method.cv
    center = method.center
    kspring = method.kspring
    natoms = np.size(snapshot.positions, 0)

    def initialize():
        bias = np.zeros((natoms, 3))
        return HarmonicBiasState(bias, None)

    def update(state, data):
        xi, Jxi = cv(data)
        forces = kspring @ (xi - center).flatten()
        bias = -Jxi.T @ forces.flatten()
        bias = bias.reshape(state.bias.shape)

        return HarmonicBiasState(bias, xi)

    return snapshot, initialize, generalize(update, helpers)
