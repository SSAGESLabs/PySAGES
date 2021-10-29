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
        dummy = np.zeros(len(cvs))
        super().__init__(cvs, dummy, dummy, args, kwargs)

    def run(self,
            context_generator: Callable,
            timesteps,
            centers,
            ksprings,
            periods,
            context_args=dict(),
            **kwargs):
        """
        Implementation of the serial execution of umbrella integration with up to linear order (ignoring second order terms with covariance matrix) as described in
        J. Chem. Phys. 131, 034109 (2009); https://doi.org/10.1063/1.3175798 Equation 13. Higher order approximations can be implemented by the user using the provided covariance matrix.

        context_generator: user defined function that sets up a simulation context with the backend.
                           Must return an instance of hoomd.conext.SimulationContext for hoomd-blue and simtk.openmm.Context.
                           The function gets context_args unpacked for additional user args.
                           For each replica along the path, the argument "replica_num" i [0, ..., N-1] is set for the context_generator to load the appropriate initial condition.
        timesteps: number of timesteps the simulation is running.
        centers: list of CV centers along the path of integration. The length defines the number replicas.
        ksprings: float or list of floats describing the spring strength of the harmonic bias for each replica.
        periods: int of list of int describing the period for the histrogram logging of each replica.
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

        result = {}
        result["histogram"] = []
        result["histogram_means"] = []
        result["kspring"] = []
        result["center"] = []
        result["nabla_A"] =  []
        result["A"] = []

        for rep in range(Nreplica):
            context_args["replica_num"] = rep
            self.center = centers[rep]
            self.kspring = ksprings[rep]
            callback = HistogramLogger(periods[rep])
            context = context_generator(**context_args)
            wrapped_context = ContextWrapper(context, self, callback)
            with wrapped_context:
                wrapped_context.run(timesteps[rep])

            result["kspring"].append(self.kspring)
            result["center"].append(self.center)

            result["histogram"].append(callback)
            result["histogram_means"].append(callback.get_means())

            # Equation 13
            result["nabla_A"].append(-result["kspring"][rep] @ (result["histogram_means"][rep] - result["center"][rep]))
            # discrete forward integration of the free-energy
            if rep == 0:
                result["A"].append(0)
            else:
                result["A"].append( result["A"][rep-1] + result["nabla_A"][rep-1].T @ (result["center"][rep] - result["center"][rep-1]))

        return result
