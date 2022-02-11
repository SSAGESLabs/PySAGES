# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES
"""
Harmonic bias method.

Biasing a simulation towards a value of a collective variable is the foundation of a
number advanced sampling methods, umbrella integration, WHAM, string method to name a few.
This method implements such a bias.

The hamiltonian is ammended with a term :math:`\\mathcal{H} = \\mathcal{H}_0 + \\mathcal{H}_\\mathrm{HB}(\\xi)` where
:math:`\\mathcal{H}_\\mathrm{HB}(\\xi) = \\boldsymbol{K}/2 (\\xi_0 - \\xi)^2` biases the simulations around the collective variable :math:`\\xi_0`.
"""


from typing import NamedTuple
import jax.numpy as np
from jaxlib.xla_extension import DeviceArray as JaxArray  # pylint: disable=no-name-in-module
from .core import SamplingMethod, generalize  # pylint: disable=relative-beyond-top-level


class HarmonicBiasState(NamedTuple):
    """
    Description of a state bias by a harmonic potential for a CV.

    bias: JaxArray
        Array with harmic biasing forces for each particle in the simulation.
    xi: JaxArray
        Collective variable value of the last simulation step.
    """
    bias: JaxArray
    xi: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class HarmonicBias(SamplingMethod):
    """
    Harmonic bias method class.
    """
    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, kspring, center, *args, **kwargs):
        """
        Arguments
        ---------
        cvs: Union[List, Tuple]
            A list or Tuple of collective variables, length `N`.
        kspring:
            A scalar, array length `N` or symmetric `N x N` matrix. Restraining spring constant.
        center:
            An array of length `N` representing the minimum of the harmonic biasing potential.
        """
        super().__init__(cvs, args, kwargs)
        self.cv_dimension = len(cvs)
        self.kspring = kspring
        self.center = center

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

    @property
    def center(self):
        """
        Retrieve current center of the collective variable.
        """
        return self._center

    @center.setter
    def center(self, center):
        """
        Set the center of the collective variable to a new position.
        """
        center = np.asarray(center)
        if center.shape == ():
            center = center.reshape(1)
        if len(center.shape) !=1 or center.shape[0] != self.cv_dimension:
            raise RuntimeError(f"Invalid center shape expected {self.cv_dimension} got {center.shape}.")
        self._center = center

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
