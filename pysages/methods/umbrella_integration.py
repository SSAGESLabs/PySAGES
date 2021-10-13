# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import jax.numpy as np
from typing import Callable
from pysages.backends import ContextWrapper
from .harmonic_bias import HarmonicBias
from .utils import HistogramLogger


class UmbrellaIntegration(HarmonicBias):
    def __init__(self, cvs, *args, **kwargs):
        super().__init__(cvs, [0], [0], args, kwargs)
        self.kspring = False
        self.center = False

    def set_center(self, center):
        self.center = np.asarray(center)

    def set_kspring(self, kspring):
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

        def listify(arg, Nreplica, name, dtype):
            if isinstance(arg, list):
                if len(arg) != Nreplica:
                    raise RuntimeError("Provided list argument {0} has not the correct length got {1}, but expected {2}".format(name, len(arg), Nreplica))
            else:
                arg = [dtype(arg) for i in range(Nreplica)]
            return arg



        Nreplica = len(centers)
        timesteps = listify(timesteps, Nreplica, "timesteps", int)
        ksprings = listify(ksprings, Nreplica, "kspring", float)
        periods = listify(periods, Nreplica, "periods", int)
        bins = listify(bins, Nreplica, "bins", int)
        ranges = listify(ranges, Nreplica, "ranges", tuple)
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
            callback = HistogramLogger(periods[rep])
            context = context_generator(**context_args)
            wrapped_context = ContextWrapper(context, self, callback)
            with wrapped_context:
                wrapped_context.run(timesteps[rep])

            histograms.append(callback.get_histograms(bins[rep], ranges[rep]))
        # missing the processing step
        return histograms
