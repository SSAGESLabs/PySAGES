# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import jax.numpy as np


class HistogramLogger:
    """
    Implements a Callback functor for methods.
    Logs the state of the collective variable to generate histograms.
    """
    def __init__(self, period : int, offset : int = 0):
        """
        Construct a logger class.
        period: timesteps between logging of collective variables.
        offset: timesteps at the beginning of a run used for equilibration
        """
        self.period = period
        self.counter = 0
        self.offset = offset
        self.data = []

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        self.counter += 1
        if self.counter > self.offset and self.counter % self.period == 0:
            self.data.append(state.xi[0])

    def get_histograms(self, **kwargs):
        """
        Helper function to generate histrograms from the collected CV data. kwargs are passed on to numpys histogramdd function.
        """
        data = np.asarray(self.data)
        if "density" not in kwargs:
            kwargs["density"] = True
        return np.histogramdd(data, **kwargs)

    def get_means(self):
        """
        Return mean values of the histogram data.
        """
        data = np.asarray(self.data)
        return np.mean(data, axis=0)

    def get_cov(self):
        """
        Return covariance matrix of the histgram data.
        """
        data = np.asarray(self.data)
        return np.cov(data.T)

    def reset(self):
        """
        Reset internal state.
        """
        self.counter = 0
        self.data = []
