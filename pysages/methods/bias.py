# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Generic abstract bias method.
"""

from abc import abstractmethod
from jax import numpy as np
from pysages.methods.core import SamplingMethod


class Bias(SamplingMethod):
    """
    Abstract biasing class.
    In this context a biasing methond ensures that a system is biased around a fixed `center` in CV space.
    How this biasing is achieved is up to the individual implementation.
    A common biasing form is implemented via the `HarmonicBias` class.

    Biasing is commonly used in other advanced sampling methods, such as UmbrellaIntegration
    or the ImprovedString method.
    This abstract class defines an interface to interact with the CV center, such that method can rely on it.
    """

    __special_args__ = {"center"}
    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, center, **kwargs):
        """
        Arguments
        ---------
        cvs: Union[List, Tuple]
            A list or tuple of collective variables, length `N`.
        center:
            An array of length `N` representing the minimum of the harmonic biasing potential.
        """
        super().__init__(cvs, **kwargs)
        self.cv_dimension = len(cvs)
        self.center = center

    def __getstate__(self):
        state, kwargs = super().__getstate__()
        state["center"] = self._center
        return state, kwargs

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
        if len(center.shape) != 1 or center.shape[0] != self.cv_dimension:
            raise RuntimeError(
                f"Invalid center shape expected {self.cv_dimension} got {center.shape}."
            )
        self._center = center

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass
