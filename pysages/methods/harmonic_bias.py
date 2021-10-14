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

    bias -- additional forces for the class.
    xi -- current cv value.
    """
    bias: JaxArray
    xi: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class HarmonicBias(SamplingMethod):
    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, kspring, center, *args, **kwargs):
        """
        cvs: list of collective variables length N
        kspring: float, array length N or symmetric NxN matrix. Restraining spring constant.
        center: array of length N representing the minimum of the harmonic draining potential.
        args and kwargs for parent constructor
        """
        super().__init__(cvs, args, kwargs)
        self._N = len(cvs)
        self.set_kspring(kspring)
        self.set_center(center)

    def set_kspring(self, kspring):
        """
        Set new spring constant.
        kspring: float, array length N or symmetric NxN matrix. Restraining spring constant.
        """

        # ensure array
        kspring = np.asarray(kspring)
        if len(kspring.shape) == 2:
            if kspring.shape != (self._N, self._N):
                raise RuntimeError("2D kspring with wrong shape, expected {0}x{0} got {1}".format(self._N, kspring.shape))
            self._kspring = kspring
        else:
            if len(kspring.shape) == 0:
                kspring = np.ones(self._N) * kspring
            if len(kspring.shape) !=1:
                raise RuntimeError("Wrong shape of kspring {0}, expected 1D or 2D".format(kspring.shape))
            if not (kspring.shape[0] == self._N or kspring.shape[0] == 1) :
                raise RuntimeError("Length of 1D kspring unequal with the length of collective variables. {0} {1}".format(kspring.shape, self._N))

            self._kspring = np.identity(self._N) * kspring

        if not np.allclose(self._kspring, self._kspring.T):
            raise RuntimeError("Spring matrix is not symmetric")

        return self._kspring

    def get_kspring(self):
        return self._kspring

    def set_center(self, center):
        center = np.asarray(center)
        if len(center.shape) !=1 or center.shape[0] != self._N:
            raise RuntimeError("Invalid center shape expected {0} got {1}.".format(self._N, center.shape))
        self._center = center


    def get_center(self):
        return self._center

    def build(self, snapshot, helpers):
        return _harmonic_bias(self, snapshot, helpers)


def _harmonic_bias(method, snapshot, helpers):
    cv = method.cv
    center = method.get_center()
    kspring = method.get_kspring()
    natoms = np.size(snapshot.positions, 0)

    def initialize():
        bias = np.zeros((natoms, 3))
        return HarmonicBiasState(bias, None)

    def update(state, rs, vms, ids):
        xi, Jxi = cv(rs, indices(ids))
        D = kspring @ (xi - center).flatten()
        bias = -Jxi.T @ D.flatten()
        bias = bias.reshape(state.bias.shape)

        return HarmonicBiasState(bias, xi)

    return snapshot, initialize, generalize(update, helpers)
