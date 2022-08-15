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

from copy import deepcopy
from typing import Callable, Optional, Union

import plum
from pysages.methods.core import Result, SamplingMethod, _run
from pysages.methods.harmonic_bias import HarmonicBias
from pysages.methods.utils import HistogramLogger, listify, SerialExecutor
from pysages.utils import dispatch


class UmbrellaIntegration(SamplingMethod):
    """
    This class combines harmonic biasing with multiple replicas.
    It also collects histograms of the collective variables throughout the simulations
    for subsequent analysis.
    By default the class also estimates an approximation of the free energy landscape
    along the given path via umbrella integration.
    Note that this is not very accurate and usually requires more sophisticated analysis on top.
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
        Initialization, sets up the HarmonicBias subsamplers.

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

        replicas = len(centers)
        ksprings = listify(ksprings, replicas, "ksprings", float)
        periods = listify(hist_periods, replicas, "hist_periods", int)
        offsets = listify(hist_offsets, replicas, "hist_offsets", int)

        self.submethods = [HarmonicBias(cvs, k, c) for (k, c) in zip(ksprings, centers)]
        self.histograms = [HistogramLogger(p, o) for (p, o) in zip(periods, offsets)]

    @plum.dispatch
    def __init__(  # noqa: F811 # pylint: disable=C0116,E0102
        self,
        biasers: list,
        hist_periods: Union[list, int],
        hist_offsets: Union[list, int] = 0,
        **kwargs
    ):
        cvs = None
        for bias in biasers:
            if cvs is None:
                cvs = bias.cvs
            else:
                if bias.cvs != cvs:
                    raise RuntimeError(
                        "Attempted run of UmbrellaSampling with different CVs"
                        " for the individual biaser."
                    )
        super().__init__(cvs, **kwargs)
        replicas = len(biasers)
        periods = listify(hist_periods, replicas, "hist_periods", int)
        offsets = listify(hist_offsets, replicas, "hist_offsets", int)

        self.submethods = biasers
        self.histograms = [HistogramLogger(p, o) for (p, o) in zip(periods, offsets)]

    # We delegate the sampling work to HarmonicBias
    # (or possibly other methods in the future)
    def build(self):  # pylint: disable=arguments-differ
        pass


@dispatch
def run(  # pylint: disable=arguments-differ
    method: UmbrellaIntegration,
    context_generator: Callable,
    timesteps: Union[int, float],
    context_args: Optional[dict] = None,
    post_run_action: Optional[Callable] = None,
    executor=SerialExecutor(),
    **kwargs
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

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.

    kwargs:
        Passed to the backend run function as additional user arguments.

    post_run_action: Optional[Callable] = None
        Callable function that enables actions after the run execution of PySAGES.
        Actions are executed inside the generated context.
        Example uses for this include writing a final configuration file.
        This function gets `context_args` unpacked just like `context_generator`.

    * Note:
        This method does not accept a user defined callback.
    """
    timesteps = int(timesteps)
    context_args = {} if context_args is None else context_args

    def submit_work(executor, method, context_args, callback):
        return executor.submit(
            _run,
            method,
            context_generator,
            timesteps,
            context_args,
            callback,
            post_run_action,
            **kwargs
        )

    futures = []
    with executor as ex:
        for rep, submethod in enumerate(method.submethods):
            local_context_args = deepcopy(context_args)
            local_context_args["replica_num"] = rep
            callback = method.histograms[rep]
            futures.append(submit_work(ex, submethod, local_context_args, callback))
    results = [future.result() for future in futures]
    states = [r.states for r in results]
    callbacks = [r.callbacks for r in results]

    return Result(method, states, callbacks)


@dispatch
def analyze(result: Result[UmbrellaIntegration]):
    """
    Computes the free energy from the result of an `UmbrellaIntegration` run.
    """

    def free_energy_gradient(k_spring, xi_mean, xi_ref):
        """
        Equation 13 from https://doi.org/10.1063/1.3175798
        """
        return -(k_spring @ (xi_mean - xi_ref))

    def integrate(A, nabla_A, centers, i):
        return A[i - 1] + nabla_A[i - 1].T @ (centers[i] - centers[i - 1])

    submethods = result.method.submethods

    ksprings = [s.kspring for s in submethods]
    centers = [s.center for s in submethods]
    hist_means = [cb.get_means() for cb in result.callbacks]

    submethods_iterator = zip(ksprings, centers, hist_means)
    mean_forces = []
    free_energy = [0.0]

    for i, (kspring, center, mean) in enumerate(submethods_iterator):
        mean_forces.append(free_energy_gradient(kspring, mean, center))
        if i > 0:
            free_energy.append(integrate(free_energy, mean_forces, centers, i))

    return dict(
        ksprings=ksprings,
        centers=centers,
        histograms=result.callbacks,
        histogram_means=hist_means,
        mean_forces=mean_forces,
        free_energy=free_energy,
    )
