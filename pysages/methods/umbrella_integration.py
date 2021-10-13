# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import jax.numpy as np
from typing import Callable
from .harmonic_bias import HarmonicBias
from pysages.backends import ContextWrapper

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

    def get_histograms(self, bins, lim):
        """
        Helper function to generate histrograms from the collected CV data.
        """
        data = np.asarray(self.data)
        data = data.reshape(data.shape[0], data.shape[2])
        histograms = []
        for i in range(data.shape[1]):
            histograms.append(np.histogram(data[:,i], bins=bins, range=lim, density=True)[0])
        return histograms

    def reset(self):
        """
        Reset internal state.
        """
        self.counter = 0
        self.data = []


class UmbrellaIntegration(HarmonicBias):
    def __init__(self, cvs, *args, **kwargs):
        super().__init__(cvs, args, kwargs)
        self.kspring = None
        self.center = None

    def set_center(center):
        self.center = np.asarray(center)

    def set_kspring(kspring):
        self.kspring = np.asarray(kspring)

    def run(self,
            context_generator: Callable,
            timesteps,
            centers,
            ksprings,
            periods,
            bins,
            ranges,
            context_args=dict(),
            **kwargs):
        """
        Implementation of the serial execution of umbrella integration as described in
        Kästner, Johannes. "Umbrella sampling." Wiley Interdisciplinary Reviews: Computational Molecular Science 1.6 (2011): 932-942.
        context_generator: user defined function that sets up a simulation context with the backend.
                           Must return an instance of hoomd.conext.SimulationContext for hoomd-blue and simtk.openmm.Context.
                           The function gets context_args unpacked for additional user args.
                           For each replica along the path, the argument "replica_num" i [0, ..., N-1] is set for the context_generator to load the appropriate initial condition.
        timesteps: number of timesteps the simulation is running.
        centers: list of CV centers along the path of integration. The length defines the number replicas.
        ksprings: float or list of floats describing the spring strength of the harmonic bias for each replica.
        periods: int of list of int describing the period for the histrogram logging of each replica.
        bins: int of list of int describing the bin size for the histrogram logging of each replica.
        ranges: (min, max) of list of (min,max) describing the histogram range of each replica.
        kwargs: gets passed to the backend run function for additional user arguments to be passed down.
        User defined callback are not available, the method requires use of a builtin callback.
        """

        Nreplica = len(centers)
        if type(timesteps) == int:
            timesteps = [timesteps for i in range(Nreplica)]
        if len(timesteps) != len(centers):
            raise RuntimeError("Provided length of centers ({0}) and timesteps ({1}) is not equal.".format(len(centers), len(timesteps)))

        if type(ksprings) == float:
            ksprings = [ksprings for i in range(Nreplica)]
        if len(ksprings) != len(centers):
            raise RuntimeError("Provided length of centers ({0}) and ksprings ({1}) is not equal.".format(len(centers), len(ksprings)))

        if type(periods) == int:
            periods = [periods for i in range(Nreplica)]
        if len(periods) != len(centers):
            raise RuntimeError("Provided length of centers ({0}) and periods ({1}) is not equal.".format(len(centers), len(periods)))

        if type(ranges) == tuple:
            ranges = [ranges for i in range(Nreplica)]
        if len(ranges) != len(centers):
            raise RuntimeError("Provided length of centers ({0}) and periods ({1}) is not equal.".format(len(centers), len(periods)))
        for hilo in ranges:
            if len(hilo) != 2:
                raise RuntimeError("Provided ranges have a different length from two.")
            if hilo[0] >= hilo[1]:
                raise RuntimeError("Provided ranges have invalid high/low values.")


        histograms = []
        for rep in range(Nreplica):
            context_args["replica_num"] = rep
            self.set_center(centers[rep])
            self.set_kspring(ksprings[rep])
            callback = HistrogramLogger(periods[rep])
            wrapped_context = ContextWrapper(context, self, callback)
            with wrapped_context:
                wrapped_context.run(timesteps[i])

            histograms.append(callback.get_histograms(bins[rep], ranges[rep]))
        #process histrograms
