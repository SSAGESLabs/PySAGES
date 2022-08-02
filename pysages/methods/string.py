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

import numpy as np
from copy import deepcopy
from typing import Callable, Optional, Union, Any
from numpy.linalg import norm

import plum
from pysages.methods.core import Result, SamplingMethod, _run
from pysages.methods.harmonic_bias import HarmonicBias
from pysages.methods.umbrella_integration import UmbrellaIntegration
from pysages.methods.utils import HistogramLogger, listify, SerialExecutor
from pysages.utils import dispatch

def _test_valid_spacing(replicas, spacing):
    if spacing is None:
        spacing = np.diff(np.linspace(0, 1, replicas))
    spacing = np.asarray(spacing)
    if len(spacing) != replicas - 1:
        RuntimeError("The provided spacing for String is not replicas - 1.")
    if np.any(spacing <=0):
        RuntimeError("Provided spacing is not a positive real number monotonic increasing {space}.")

    # Normalize
    spacing /= np.sum(spacing)

    return spacing

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
        metric: Callable[[Any, Any], float] = lambda x, y: norm(x-y),
        spacing: Union[list, None] = None,
        **kwargs
    ):
        """
        Initialization, sets up the UmbrellaSampling with Harmonic bias subsamplers.
        Additional agruments are to define the execution of the string method.

        Arguments
        ---------
        centers: list[numbers.Real]
            CV inital centers along the path of integration. Its length defines the number replicas.

        ksprings: Union[float, list[float]]
            Spring constants of the harmonic biasing potential for each replica.

        hist_periods: Union[int, list[int]]
            Indicates the period for logging the CV into the histogram of each replica.

        hist_offsets: Union[int, list[int]] = 0
            Offset before starting logging into each replica's histogram.

        metric: Callable[[Any, Any], float] = lambda x, y: numpy.norm(x-y)
            Metric defining how distance is defined between points on the string.
            Defaults to the L2 norm of the difference between points in CV space.

        spacing: Union[list, None] = None
            Desired spacing between points on the string. So the length must be replicas -1.
            Note that we internally normalize the spacing [0,1].
            Defaults to equal spacing on the string.
        """

        super().__init__(cvs, **kwargs)
        replicas = len(centers)
        self.umbrella_sampler = UmbrellaIntegration(
            cvs, ksprings, centers, hist_periods, hist_offsets
        )
        self.metric = metric
        self.spacing = _test_valid_spacing(replicas, spacing)

    @plum.dispatch
    def __init__(
            self,
            umbrella_sampler: UmbrellaIntegration,
            metric: Callable[[Any, Any], float] = lambda x, y: norm(x-y),
            spacing: Union[list, None] = None,
            **kwargs
    ):
        """
        Initialization, sets up the UmbrellaSampling with Harmonic bias subsamplers.

        Arguments
        ---------
        umbrella_sampler: UmbrellaIntegration
            Sub-method that performs the biased simulations along the path in between updates.

        metric: Callable[[Any, Any], float] = lambda x, y: numpy.norm(x-y)
            Metric defining how distance is defined between points on the string.
            Defaults to the L2 norm of the difference between points in CV space.

        spacing: Union[list, None] = None
            Desired spacing between points on the string. So the length must be replicas -1.
            Note that we internally normalize the spacing [0,1].
            Defaults to equal spacing on the string.
        """

        super().__init__(umbrella_sampler.cvs, **kwargs)
        replicas = len(umbrella_sampler.submethods)
        self.umbrella_sampler = umbrella_sampler
        self.metric = metric
        self.spacing = _test_valid_spacing(replicas, spacing)


    # We delegate the sampling work to UmbrellaIntegration
    def build(self):  # pylint: disable=arguments-differ
        pass


@dispatch
def run(  # pylint: disable=arguments-differ
    method: ImprovedString,
    context_generator: Callable,
    timesteps: Union[int, float],
    stringsteps: Union[int, float],
    context_args: Optional[dict] = None,
    post_run_action: Optional[Callable] = None,
    executor=SerialExecutor(),
    **kwargs
):
    """
    Implementation of the improved string method.
    The
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
