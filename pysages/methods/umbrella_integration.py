# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Umbrella Integration.

Umbrella integration uses multiple replicas placed along a pathway in the free energy
landscape by Harmonic Bias simulations.
From the statistics of the simulations, the thermodynamic forces along the path are
determined and integrated to obtain an approximation of the free energy landscape.
This class implements the replica simulations and approximates the free energy.
However, the method is not very accurate and it is preferred that more advanced methods
(e.g. the Weighted Histogram Analysis Method) are used for the analysis of the simulations.
"""

from typing import Callable, Optional

from jax import numpy as np

from pysages.backends import ContextWrapper
from pysages.methods.harmonic_bias import HarmonicBias
from pysages.methods.utils import HistogramLogger
from pysages.utils import dispatch


class UmbrellaIntegration(HarmonicBias):
    """
    Umbrella Integration class.

    This class combines harmonic biasing with multiple replicas.
    It also collects histograms of the collective variables throughout the simulations
    for subsequent analysis.
    By default the class also estimates an approximation of the free energy landscape
    along the given path via umbrella integration.
    Note that this is not very accurate and ususally requires more sophisticated analysis on top.
    """

    def __init__(self, cvs, *args, **kwargs):
        """
        Initialization, mostly defining the collective variables and setting up the
        underlying Harmonic Bias.
        """
        kspring = center = np.zeros(len(cvs))
        super().__init__(cvs, kspring, center, *args, **kwargs)


@dispatch
def run(  # pylint: disable=arguments-differ
    method: UmbrellaIntegration,
    context_generator: Callable,
    timesteps: int,
    centers,
    ksprings,
    hist_periods,
    hist_offsets=0,
    context_args: Optional[dict] = None,
    **kwargs,
):
    """
    Implementation of the serial execution of umbrella integration with up to linear
    order (ignoring second order terms with covariance matrix) as described in
    J. Chem. Phys. 131, 034109 (2009); https://doi.org/10.1063/1.3175798 (equation 13).
    Higher order approximations can be implemented by the user using the provided
    covariance matrix.

    Arguments
    ---------
    context_generator: Callable
        User defined function that sets up a simulation context with the backend.
        Must return an instance of `hoomd.conext.SimulationContext` for HOOMD-blue and
        `openmm.Context` for OpenMM.
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

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.

    kwargs:
        Passed to the backend run function as additional user arguments.

    * Note:
        This method does not accept a user defined callback.
    """

    def free_energy_gradient(k_spring_tensor, mean, center):
        """
        Equation 13 from https://doi.org/10.1063/1.3175798
        """
        return -(k_spring_tensor @ (mean - center))

    def integrate(a_free_energy, nabla_a_free_energy, centers, i):
        return a_free_energy[i - 1] + nabla_a_free_energy[i - 1].T @ (centers[i] - centers[i - 1])

    def collect(arg, n_replica, name, dtype):
        if isinstance(arg, list):
            if len(arg) != n_replica:
                raise RuntimeError(
                    f"Invalid length for argument {name} " f"(got {len(arg)}, expected {n_replica})"
                )
        else:
            arg = [dtype(arg) for i in range(n_replica)]
        return arg

    context_args = {} if context_args is None else context_args

    n_replica = len(centers)
    timesteps = collect(timesteps, n_replica, "timesteps", int)
    ksprings = collect(ksprings, n_replica, "kspring", float)
    hist_periods = collect(hist_periods, n_replica, "hist_periods", int)
    hist_offsets = collect(hist_offsets, n_replica, "hist_offsets", int)

    result = {}
    result["histogram"] = []
    result["histogram_means"] = []
    result["kspring"] = []
    result["center"] = []
    result["nabla_a_free_energy"] = []
    result["a_free_energy"] = []

    states = []

    for rep in range(n_replica):
        method.center = centers[rep]
        method.kspring = ksprings[rep]

        context_args["replica_num"] = rep
        context = context_generator(**context_args)
        callback = HistogramLogger(hist_periods[rep], hist_offsets[rep])

        wrapped_context = ContextWrapper(context, method, callback)
        with wrapped_context:
            wrapped_context.run(timesteps[rep], **kwargs)  # pylint: disable=E1102

        states.append(wrapped_context.sampler.state)

        mean = callback.get_means()

        result["kspring"].append(method.kspring)
        result["center"].append(method.center)
        result["histogram"].append(callback)
        result["histogram_means"].append(mean)
        result["nabla_a_free_energy"].append(
            free_energy_gradient(method.kspring, mean, method.center)
        )
        # Discrete forward integration of the free-energy
        if rep == 0:
            result["a_free_energy"].append(0)
        else:
            result["a_free_energy"].append(
                integrate(
                    result["a_free_energy"],
                    result["nabla_a_free_energy"],
                    result["center"],
                    rep,
                )
            )

    return result
