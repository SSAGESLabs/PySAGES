# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import jax.numpy as np



class HistogramLogger:
    """
    Implements a Callback functor for methods.
    Logs the state of the collective variable to generate histograms.
    """
    def __init__(self, period : int):
        """
        Construct a logger class.
        period: timesteps between logging of collective variables.
        """
        self.period = period
        self.counter = 0
        self.data = []

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        self.counter += 1
        if self.counter % self.period == 0:
            self.data.append(state.xi)

    def get_histograms(self, bins, lims):
        """
        Helper function to generate histrograms from the collected CV data.
        """
        data = np.asarray(self.data)
        data = data.reshape(data.shape[0], data.shape[2])
        histograms, edges = np.histogramdd(data, bins=bins, range=lims, density=True)
        return histograms, edges

    def reset(self):
        """
        Reset internal state.
        """
        self.counter = 0
        self.data = []
