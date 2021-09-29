# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import NamedTuple
from .core import SamplingMethod, generalize  # pylint: disable=relative-beyond-top-level
import jax.numpy as np
from jaxlib.xla_extension import DeviceArray as JaxArray


class UmbrellaState(NamedTuple):
    """
    Description of an umbrella sampling state.

    bias -- additional forces for the class.
    xi -- current cv value.
    """
    bias: JaxArray
    xi: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class UmbrellaSampling(SamplingMethod):
    def __init__(self, cvs, kspring, center, *args, **kwargs):
        super().__init__(cvs, args, kwargs)
        self.kspring = np.asarray(kspring)
        self.center = np.asarray(center)

    def __call__(self, snapshot, helpers):
        return _umbrella(snapshot, self.cv, self.center, self.kspring, helpers)


def _umbrella(snapshot, cv, center, kspring, helpers):
    natoms = np.size(snapshot.positions, 0)
    indices = helpers.indices

    def initialize():
        bias = np.zeros((natoms, 3))
        return UmbrellaState(bias, None)

    def update(state, rs, vms, ids):
        xi, Jxi = cv(rs, indices(ids))
        D = kspring * (xi - center)
        bias = (-D * Jxi).reshape(state.bias.shape)
        return UmbrellaState(bias, xi)

    return snapshot, initialize, generalize(update)
