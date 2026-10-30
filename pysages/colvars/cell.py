# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective variables of the simulation box (they need a constant-pressure simulation).
"""

from jax import numpy as np

from pysages.colvars.core import CollectiveVariable, UInt32


class BoxVolume(CollectiveVariable):
    """
    Volume of the simulation box, ``det(H)``.
    """

    requires_box = True

    def __init__(self):  # pylint: disable=super-init-not-called
        self.indices = np.zeros(0, dtype=UInt32)
        self.groups = []
        self.requires_box_unwrapping = False

    @property
    def function(self):
        return np.linalg.det
