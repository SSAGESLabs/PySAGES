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
        super().__init__(cvs, args, kwargs)
        self.kspring = np.asarray(kspring)
        self.center = np.asarray(center)

    def build(self, snapshot, helpers):
        return _harmonic_bias(self, snapshot, helpers)


def _harmonic_bias(method, snapshot, helpers):
    cv = method.cv
    center = method.center.reshape(1, -1)
    kspring = method.kspring.reshape(1, -1)
    natoms = np.size(snapshot.positions, 0)
    indices = helpers.indices

    def initialize():
        bias = np.zeros((natoms, 3))
        return HarmonicBiasState(bias, None)

    def update(state, rs, vms, ids):
        xi, Jxi = cv(rs, indices(ids))
        D = kspring * (xi - center)
        bias = -Jxi.T @ D.flatten()
        bias = bias.reshape(state.bias.shape)

        return HarmonicBiasState(bias, xi)

    return snapshot, initialize, generalize(update)
