# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collection of helpful classes for methods.

This includes callback functors (callable classes).
"""

import jax.numpy as np
from jax import lax


class HistogramLogger:
    """
    Implements a Callback functor for methods.
    Logs the state of the collective variable to generate histograms.
    """

    def __init__(self, period: int, offset: int = 0):
        """
        HistogramLogger constructor.

        Arguments
        ---------
        period:
            Timesteps between logging of collective variables.

        offset:
            Timesteps at the beginning of a run used for equilibration.
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
        Helper function to generate histrograms from the collected CV data.
        `kwargs` are passed on to `numpy.histogramdd` function.
        """
        data = np.asarray(self.data)
        if "density" not in kwargs:
            kwargs["density"] = True
        return np.histogramdd(data, **kwargs)

    def get_means(self):
        """
        Returns mean values of the histogram data.
        """
        data = np.asarray(self.data)
        return np.mean(data, axis=0)

    def get_cov(self):
        """
        Returns covariance matrix of the histgram data.
        """
        data = np.asarray(self.data)
        return np.cov(data.T)

    def reset(self):
        """
        Reset internal state.
        """
        self.counter = 0
        self.data = []


# callback to log hills and other output files in metadynamics
# NOTE: for OpenMM; issue #16 on openmm-dlext should be resolved for this to work properly.
class MetaDLogger:
    """
    Implements a Callback functor for methods.
    Logs the state of the collective variable and other parameters in metadynamics.
    """

    def __init__(self, stride, sigma, height, hillsFile):
        """
        logMeta constructor.

        Arguments
        ---------
        hills_period:
            Timesteps between logging of collective variables and metadynamics parameters.

        sigma:
            Width of the Gaussian bias potential.

        height:
            Height of the Gaussian bias potential.

        hillsFile:
            Name of the output hills log file.

        counter:
            Local frame counter.
        """
        self.stride = stride
        self.sigma = sigma
        self.height = height
        self.hillsFile = hillsFile
        self.counter = 0

    # write hills file
    def write_hills_to_file(self, xi, sigma, height):
        """
        Append the centers, standard deviations and heights to log file.
        """
        with open(self.hillsFile, "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            for j in range(xi.shape[0]):
                f.write(str(xi[j]) + "\t")
            for j in range(sigma.shape[0]):
                f.write(str(sigma[j]) + "\t")
            f.write(str(height) + "\n")

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """

        local_loop = lax.select(state.idx == 0, 0, state.idx - 1)
        # Write hills file containing CV centers and corresponding heights
        if self.counter != 0 and self.counter % self.stride == 0:
            self.write_hills_to_file(
                state.xis[local_loop], state.sigmas[local_loop], state.heights[local_loop]
            )

        self.counter += 1
