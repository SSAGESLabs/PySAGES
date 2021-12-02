# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import Callable

from pysages.backends import ContextWrapper
from pysages.methods.harmonic_bias import HarmonicBias
from pysages.methods.utils import HistogramLogger

import jax.numpy as np


class UmbrellaIntegration(HarmonicBias):
    def __init__(self, cvs, *args, **kwargs):
        kspring = center = np.zeros(len(cvs))
        super().__init__(cvs, kspring, center, args, kwargs)

    def run(
        self,
        context_generator: Callable,
        timesteps: int,
        centers,
        ksprings,
        hist_periods,
        hist_offsets = 0,
        context_args = dict(),
        **kwargs
    ):
        """
        Implementation of the serial execution of umbrella integration
        with up to linear order (ignoring second order terms with covariance matrix)
        as described in J. Chem. Phys. 131, 034109 (2009); https://doi.org/10.1063/1.3175798 (equation 13).
        Higher order approximations can be implemented by the user using the provided covariance matrix.

        Arguments
        ---------
        context_generator: Callable
            User defined function that sets up a simulation context with the backend.
            Must return an instance of hoomd.conext.SimulationContext for HOOMD-blue and openmm.Context for OpenMM.
            The function gets `context_args` unpacked for additional user args.
            For each replica along the path, the argument `replica_num` in [0, ..., N-1]
            is set in the `context_generator` to load the appropriate initial condition.

        timesteps: int
            Number of timesteps the simulation is running.

        centers: list[numbers.Real]
            CV centers along the path of integration. The length defines the number replicas.

        ksprings: Union[float, list[float]]
            Spring constants of the harmonic biasing potential for each replica.

        hist_periods: Union[int, list[int]]
            Describes the period for the histrogram logging of each replica.

        hist_offsets: Union[int, list[int]]
            Offset applied before starting the histogram of each replica.

        kwargs:
            Passed to the backend run function as additional user arguments.

        * Note:
            This method does not accepts a user defined callback are not available.
        """

        def free_energy_gradient(K, mean, center, i):
            "Equation 13 from https://doi.org/10.1063/1.3175798"
            return -(K @ (mean - center))

        def integrate(A, nabla_A, centers, i):
            return A[i-1] + nabla_A[i-1].T @ (centers[i] - centers[i-1])

        def collect(arg, Nreplica, name, dtype):
            if isinstance(arg, list):
                n = len(arg)
                if n != Nreplica:
                    raise RuntimeError(f"Provided list argument {name} has not the correct length (got {n}, expected {Nreplica})")
            else:
                arg = [dtype(arg) for i in range(Nreplica)]
            return arg

        Nreplica = len(centers)
        timesteps = collect(timesteps, Nreplica, "timesteps", int)
        ksprings = collect(ksprings, Nreplica, "kspring", float)
        hist_periods = collect(hist_periods, Nreplica, "hist_periods", int)
        hist_offsets = collect(hist_offsets, Nreplica, "hist_offsets", int)

        result = {}
        result["histogram"] = []
        result["histogram_means"] = []
        result["kspring"] = []
        result["center"] = []
        result["nabla_A"] =  []
        result["A"] = []

        for rep in range(Nreplica):
            self.center = centers[rep]
            self.kspring = ksprings[rep]

            context_args["replica_num"] = rep
            context = context_generator(**context_args)
            callback = HistogramLogger(hist_periods[rep], hist_offsets[rep])
            wrapped_context = ContextWrapper(context, self, callback)

            with wrapped_context:
                wrapped_context.run(timesteps[rep])

            mean = callback.get_means()

            result["kspring"].append(self.kspring)
            result["center"].append(self.center)
            result["histogram"].append(callback)
            result["histogram_means"].append(mean)
            result["nabla_A"].append(free_energy_gradient(self.kspring, mean, self.center))
            # Discrete forward integration of the free-energy
            if rep == 0:
                result["A"].append(0)
            else:
                result["A"].append(integrate(result["A"], result["nabla_A"], result["center"], rep))

        return result
