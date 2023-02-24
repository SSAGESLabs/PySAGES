# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Spline String method (aka Improved String method).
The improved string methods finds the local minimum free-energy path (MFEP)
trough a given landscape.
It uses umbrella integration to advance the discretized replicas in the CV space.
A spline interpolation helps to (equally) space the CV points along the string.
The process converges if the MFEP is found.
The final free-energy profile is calculated the same way as in UmbrellaIntegration.
We aim to implement this:
`Weinan, E., et. al. J. Chem. Phys. 126.16 (2007): 164103 <https://doi.org/10.1063/1.2720838>`_.
"""

from typing import Callable, List, Optional, Union

import numpy as np
from numpy.linalg import norm
from scipy.interpolate import interp1d

import pysages
from pysages.methods.core import Result, SamplingMethod
from pysages.methods.umbrella_integration import UmbrellaIntegration
from pysages.methods.utils import SerialExecutor, listify, numpyfy_vals
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
    spacing = np.asarray([0] + list(spacing))
    spacing = np.cumsum(spacing)
    assert abs(spacing[0]) < 1e-6
    assert abs(spacing[-1] - 1) < 1e-6
    spacing[0] = 0.0
    spacing[-1] = 1.0

    return spacing


class SplineString(SamplingMethod):
    """
    This class combines UmbrellaIntegration of multiple replicas with the path evolution
    of the spline (improved) string method. It also collects histograms of the collective variables
    throughout the simulations for subsequent analysis.
    By default the class also estimates an approximation of the free energy landscape
    along the given path via umbrella integration.
    """

    @dispatch
    def __init__(
        self,
        cvs,
        ksprings,
        centers,
        alpha,
        hist_periods: Union[List, int],
        hist_offsets: Union[List, int] = 0,
        metric: Callable = lambda x, y: norm(x - y),
        spacing: Union[List, None] = None,
        freeze_idx: List = [],
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

        alpha: Union[float, List[float]],
            Update step size for the spring. New string positions are updated according to
            :math:`\\xi_{i+1} = \\alpha (\\langle \\xi\\rangle - \\xi_i)/|\\langle \\xi\\rangle - \\xi_i|`.  # noqa=E501

        hist_periods: Union[int, List[int]]
            Indicates the period for logging the CV into the histogram of each replica.

        hist_offsets: Union[int, List[int]] = 0
            Offset before starting logging into each replica's histogram.

        metric: Callable[[JaxArray, JaxArray], float] = lambda x, y: numpy.norm(x-y)
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
        self.alpha = listify(alpha, np.asarray(self.cvs).size, "alpha", float)
        self.metric = metric
        self.spacing = _test_valid_spacing(replicas, spacing)
        self.freeze_idx = freeze_idx
        self.path_history = []

    @dispatch
    def __init__(  # noqa: F811 # pylint: disable=C0116,E0102
        self,
        umbrella_sampler: UmbrellaIntegration,
        alpha,
        metric: Callable = lambda x, y: norm(x - y),
        spacing: Union[List, None] = None,
        freeze_idx: List = [],
        **kwargs
    ):
        """
        Initialization, sets up the UmbrellaSampling with Harmonic bias subsamplers.

        Arguments
        ---------
        umbrella_sampler: UmbrellaIntegration
            Sub-method that performs the biased simulations along the path in between updates.

        metric: Callable[[JaxArray, JaxArray], float] = lambda x, y: numpy.norm(x-y)
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
        self.alpha = np.asarray(listify(alpha, np.asarray(self.cvs).size, "alpha", float))
        self.metric = metric
        self.spacing = _test_valid_spacing(replicas, spacing)
        self.freeze_idx = freeze_idx
        self.path_history = []

    # We delegate the sampling work to UmbrellaIntegration
    def build(self):  # pylint: disable=arguments-differ
        pass


@dispatch
def run(  # pylint: disable=arguments-differ
    method: SplineString,
    context_generator: Callable,
    timesteps: Union[int, float],
    stringsteps: Union[int, float],
    context_args: Optional[dict] = None,
    post_run_action: Optional[Callable] = None,
    executor=SerialExecutor(),
    executor_shutdown=True,
    **kwargs
):
    """
    Implementation of the spline interpolated (improved) string method.

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
        context_args["stringstep"] = len(method.path_history)
        # Reset histograms before new simulation
        for i in range(len(method.umbrella_sampler.histograms)):
            if i not in method.freeze_idx:
                method.umbrella_sampler.histograms[i].reset()
        umbrella_result = pysages.run(
            method.umbrella_sampler,
            context_generator,
            timesteps,
            context_args,
            post_run_action,
            executor,
            executor_shutdown=False,
            **kwargs
        )

        new_xi = []
        for i in range(len(umbrella_result.callbacks)):
            sampled_xi = umbrella_result.callbacks[i].get_means()
            old_xi = method.umbrella_sampler.submethods[i].center
            direction = (sampled_xi - old_xi) / method.metric(sampled_xi, old_xi)
            new_xi.append(np.asarray(old_xi) + method.alpha * np.asarray(direction))
        new_xi = np.asarray(new_xi)
        new_spacing = [0]
        for i in range(len(new_xi) - 1):
            new_spacing.append(method.metric(new_xi[i], new_xi[i + 1]))
        new_spacing = np.asarray(new_spacing)
        # Normalize
        new_spacing /= np.sum(new_spacing)
        new_spacing = np.cumsum(new_spacing)
        assert abs(new_spacing[0]) < 1e-6
        assert abs(new_spacing[-1] - 1) < 1e-6
        new_spacing[0] = 0.0
        new_spacing[-1] = 1.0

        # Transform into (Nreplica, X) shape for interpolation
        transformed_xi = new_xi.reshape((len(new_xi), np.sum(cv_shape)))
        # Interpolate path with splines

        interpolator = interp1d(new_spacing, transformed_xi, kind="cubic", axis=0)
        new_centers = []
        for i in range(len(new_spacing)):
            if i not in method.freeze_idx:
                new_centers.append(interpolator(method.spacing[i]).reshape(cv_shape))
            else:
                new_centers.append(method.umbrella_sampler.submethods[i].center)

            method.umbrella_sampler.submethods[i].center = new_centers[-1]

        method.path_history.append(new_centers)

    if executor_shutdown:
        executor.shutdown()

    return Result(
        method, umbrella_result.states, umbrella_result.callbacks, umbrella_result.snapshots
    )


@dispatch
def analyze(result: Result[SplineString]):
    umbrella_result = Result(
        result.method.umbrella_sampler, result.states, result.callbacks, result.snapshots
    )
    path_history = result.method.path_history
    ana = pysages.analyze(umbrella_result)
    ana["path_history"] = path_history
    path = []
    point_convergence = []
    for i in range(len(path_history[-1])):
        a = path_history[-2][i]
        b = path_history[-1][i]
        point_convergence.append(result.method.metric(a, b))
        path.append(a)
    ana["point_convergence"] = point_convergence
    ana["path"] = path

    return numpyfy_vals(ana)
