# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import NamedTuple
from .core import SamplingMethod, generalize  # pylint: disable=relative-beyond-top-level
import jax.numpy as np
from jaxlib.xla_extension import DeviceArray as JaxArray


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
        self._N = len(cvs)
        self.kspring = kspring
        self.center = center

    @property
    def kspring(self):
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
        N = self._N

        if len(shape) > 2:
            raise RuntimeError(f"Wrong kspring shape {shape} (expected scalar, 1D or 2D)")
        elif len(shape) == 2:
            if shape != (N, N):
                raise RuntimeError(f"2D kspring with wrong shape, expected ({N}, {N}), got {shape}")
            if not np.allclose(kspring, kspring.T):
                raise RuntimeError("Spring matrix is not symmetric")

            self._kspring = kspring
        else:  # len(shape) == 0 or len(shape) == 1
            n = kspring.size
            if n != N and n != 1:
                raise RuntimeError(f"Wrong kspring size, expected 1 or {N}, got {n}.")

            self._kspring = np.identity(N) * kspring
        return self._kspring

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center):
        center = np.asarray(center)
        if center.shape == ():
            center = center.reshape(1)
        if len(center.shape) !=1 or center.shape[0] != self._N:
            raise RuntimeError(f"Invalid center shape expected {self._N} got {center.shape}.")
        self._center = center

    def build(self, snapshot, helpers):
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
        D = kspring @ (xi - center).flatten()
        bias = -Jxi.T @ D.flatten()
        bias = bias.reshape(state.bias.shape)

        return HarmonicBiasState(bias, xi)

    return snapshot, initialize, generalize(update, helpers)
