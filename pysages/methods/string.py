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

import plum
import numpy as np
from copy import deepcopy
from typing import Callable, Optional, Union, Any, List
from numpy.linalg import norm
from scipy.interpolate import interp1d

import pysages
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
    if np.any(spacing <= 0):
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
        hist_periods: Union[List, int],
        hist_offsets: Union[List, int] = 0,
        metric: Callable[[Any, Any], float] = lambda x, y: norm(x - y),
        spacing: Union[List, None] = None,
        freeze_idx: List[int] = [],
        **kwargs
    ):
        """
        Initialization, sets up the UmbrellaSampling with Harmonic bias subsamplers.
        Additional agruments are to define the execution of the string method.

        Arguments
        ---------
        centers: List[numbers.Real]
            CV inital centers along the path of integration. Its length defines the number replicas.

        ksprings: Union[float, List[float]]
            Spring constants of the harmonic biasing potential for each replica.

        hist_periods: Union[int, List[int]]
            Indicates the period for logging the CV into the histogram of each replica.

        hist_offsets: Union[int, List[int]] = 0
            Offset before starting logging into each replica's histogram.

        metric: Callable[[Any, Any], float] = lambda x, y: numpy.norm(x-y)
            Metric defining how distance is defined between points on the string.
            Defaults to the L2 norm of the difference between points in CV space.

        spacing: Union[List, None] = None
            Desired spacing between points on the string. So the length must be replicas -1.
            Note that we internally normalize the spacing [0,1].
            Defaults to equal spacing on the string.

        freeze_idx: List[int] = []
            Index of string points that are not updated during integration.
            Defaults to an empty list, leaving the end-points open free to move to the next minimum.
            If you don't want the end-points updated provide `[0, N-1]`.
        """

        super().__init__(cvs, **kwargs)
        replicas = len(centers)
        self.umbrella_sampler = UmbrellaIntegration(
            cvs, ksprings, centers, hist_periods, hist_offsets
        )
        self.metric = metric
        self.spacing = _test_valid_spacing(replicas, spacing)
        self.freeze_idx = freeze_idx
        self.last_centers = None

    @plum.dispatch
    def __init__(
        self,
        umbrella_sampler: UmbrellaIntegration,
        metric: Callable[[Any, Any], float] = lambda x, y: norm(x - y),
        spacing: Union[List, None] = None,
        freeze_idx: List[int] = [],
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

        spacing: Union[List, None] = None
            Desired spacing between points on the string. So the length must be replicas -1.
            Note that we internally normalize the spacing [0,1].
            Defaults to equal spacing on the string.

        freeze_idx: List[int] = []
            Index of string points that are not updated during integration.
            Defaults to an empty list, leaving the end-points open free to move to the next minimum.
            If you don't want the end-points updated provide `[0, N-1]`.
        """

        super().__init__(umbrella_sampler.cvs, **kwargs)
        replicas = len(umbrella_sampler.submethods)
        self.umbrella_sampler = umbrella_sampler
        self.metric = metric
        self.spacing = _test_valid_spacing(replicas, spacing)
        self.freeze_idx = freeze_idx
        self.last_centers = None

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

    stringsteps: int
       Number of steps the string positions are iterated.
       It is the user responsibility to ensure final convergence.

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
    stringsteps = int(stringsteps)
    context_args = {} if context_args is None else context_args
    cv_shape = np.asarray(method.cvs).shape

    for step in range(stringsteps):
        context_args["stringstep"] = step
        umbrella_result = pysages.run(
            method.umbrella_sampler,
            context_generator,
            timesteps,
            context_args,
            post_run_action,
            executor,
            **kwargs
        )

        sampled_xi = [cb.get_mean().reshape(cv_shape) for cb in umbrella_result["callbacks"]]
        sampled_spacing = []
        for i in range(sampled_xi - 1):
            sampled_spacing.append(method.metric(sampled_xi[i], sampled_xi[i + 1]))
        sampled_spacing = np.asarray(sampled_spacing)
        # Normalize
        sampled_spacing /= np.sum(sampled_spacing)

        # Transform into (Nreplica, X) shape for interpolation
        transformed_xi = sampled_xi.reshape((len(sampled_xi), np.sum(cv_shape)))
        # Interpolate path with splines
        interpolator = interp1d(sampled_spacing, transformed_xi, kind="cubic")
        new_centers = []
        s = 0
        for i in range(len(sampled_spacing)):
            if i not in method.freeze_idx:
                new_centers.append(interpolator(s).reshape(cv_shape))
                # Only reset changing centers, for better statistic otherwise.
                method.umbrella_sampler.histrograms.reset()
            else:
                new_centers.append(method.umbrella_sampler.submethods[i].center)

            method.umbrella_sampler.submethods[i].center = new_centers[-1]
            s += method.spacing[i]
        assert abs(s - 1) < 1e-5

        method.last_centers = sampled_xi
        string_result = Result(method, umbrella_result["states"], umbrella_result["callbacks"])
    return string_result


@dispatch
def analyze(result: Result[ImprovedString]):

    umbrella_result = Result(result.method.umbrella_sampler, result.states, result.callbacks)
    ana = pysages.analyze(umbrella_result)
    ana["last_path"] = result.method.last_centers
    path = []
    point_convergence = []
    for i in range(len(result.method.last_centers)):
        a = result.callbacks[i].get_mean()
        b = result.method.last_centers
        point_convergence.append(result.method.metric(a, b))
        path.append(a)
    ana["point_convergence"] = np.asarray(point_convergence)
    ana["path"] = np.asarray(path)

    return ana
