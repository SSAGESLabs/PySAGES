# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from .core import SamplingMethod, generalize
from collections import namedtuple
import jax.numpy as np


class UmbrellaState(namedtuple(
        """
        Description of an umbrella sampling state.
        kspring: spring constant of the collective variable.
        cv_target: target of collective variable.
        """
        "UmbrellaState",
        (
            "bias",
            "cv_history",
        ),
        )):
    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class UmbrellaSampling(SamplingMethod):
    def __call__(self, snapshot, helpers):
        center = self.kwargs.get("center")
        kspring = self.kwargs.get("kspring")
        # catching user input smh

        return _umbrella(snapshot, self.cv, center, kspring, helpers)


def _umbrella(snapshot, cv, center, kspring, helpers):
    natoms = np.size(snapshot.positions, 0)
    indices = helpers.indices

    def initialize():
        cv_history = np.zeros((0, cv.shape))
        bias = np.zeros((natoms, 3))
        return UmbrellaState(bias, cv_history)

    def update(state, rs, vms, ids):
        xi, Jxi = cv(rs, indices(ids))
        D = kspring * (xi - center)
        bias = -D*Jxi
        cv_history = np.stack((state.cv_history, xi))
        return UmbrellaState(bias, cv_history)
    return snapshot, initialize, generalize(update)
