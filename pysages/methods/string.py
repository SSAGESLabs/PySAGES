# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Improved String method.

The improved string methods finds the local minimum free-energy path (MFEP) trough a given landscape.
It uses umbrella integration to advance the discretized replicas in the CV space.
A spline interpolation helps to (equally) space the CV points along the string.
The process converges if the MFEP is found.
The final free-energy profile is calculated the same way as in UmbrellaIntegration.
"""

from copy import deepcopy
from typing import Callable, Optional, Union

import plum
from pysages.methods.core import Result, SamplingMethod, _run
from pysages.methods.harmonic_bias import HarmonicBias
from pysages.methods.umbrella_integration import UmbrellaIntegration
from pysages.methods.utils import HistogramLogger, listify, SerialExecutor
from pysages.utils import dispatch


class ImprovedString(SamplingMethod):
    """
    This class combines UmbrellaIntegration of multiple replicas with the path evolution
    of the improved string method. It also collects histograms of the collective variables
    throughout the simulations for subsequent analysis.
    By default the class also estimates an approximation of the free energy landscape
    along the given path via umbrella integration.
    """

    @plum.dispatch
    def __init__(
        self,
        cvs,
        ksprings,
        centers,
        hist_periods: Union[list, int],
        hist_offsets: Union[list, int] = 0,
        **kwargs
    ):
        """
        Initialization, sets up the UmbrellaSampling with Harmonic bias subsamplers.

        Arguments
        ---------
        centers: list[numbers.Real]
            CV centers along the path of integration. Its length defines the number replicas.

        ksprings: Union[float, list[float]]
            Spring constants of the harmonic biasing potential for each replica.

        hist_periods: Union[int, list[int]]
            Indicates the period for logging the CV into the histogram of each replica.

        hist_offsets: Union[int, list[int]] = 0
            Offset before starting logging into each replica's histogram.
        """

        super().__init__(cvs, **kwargs)

        self.umbrella_sampler = UmbrellaIntegration(cvs, ksprings, centers, hist_periods, hist_offsets)


    # We delegate the sampling work to HarmonicBias
    # (or possibly other methods in the future)
    def build(self):  # pylint: disable=arguments-differ
        pass
