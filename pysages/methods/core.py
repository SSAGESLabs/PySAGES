# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABCMeta, abstractmethod
from functools import reduce
from inspect import getfullargspec
from operator import or_
from sys import modules as sys_modules

from jax import jit
from plum import parametric

from pysages.backends import SamplingContext
from pysages.colvars.core import build
from pysages.grids import Grid, build_grid, get_info
from pysages.methods.restraints import canonicalize
from pysages.methods.utils import ReplicasConfiguration
from pysages.typing import Callable, Optional, Union
from pysages.utils import (
    ToCPU,
    copy,
    device_platform,
    dispatch,
    dispatch_table,
    has_method,
    identity,
)

#  Base Classes
#  ============


@parametric
class Result:
    """
    Bundles the essential information needed to estimate the free energy of a
    system, as well as the necessary data for restarting the simulation.

    Parameters
    ----------
    method: ``SamplingMethod``
        The sampling method used.
    states: ``List[SamplingMethodState]``
        The last state of the sampling method for each system replica run.
    callbacks: ``Optional[List[Callback]]``
        Optional set of callbacks for all simulation replicas.
    snapshots: ``List[Snapshot]``
        Last snapshot of each replica of the simulation.
    """

    @classmethod
    def __infer_type_parameter__(cls, method, *_):
        return type(method)

    def __init__(self, method, states, callbacks, snapshots):
        self.method = method
        self.states = states
        self.callbacks = callbacks
        self.snapshots = snapshots


class ReplicaResult(Result):
    pass


class SamplingMethodMeta(ABCMeta):
    """
    Metaclass for enhanced sampling methods.

    It helps making parametric Result types serializable.
    """

    def __new__(cls, name, bases, namespace):
        S = super().__new__(cls, name, bases, namespace)
        T = Result[S]
        T.__qualname__ = T.__name__ = f"Result[{S.__name__}]"
        setattr(sys_modules[T.__module__], T.__name__, T)
        return S


class SamplingMethod(metaclass=SamplingMethodMeta):
    """
    Abstract base class for all sampling methods.

    This class defines a constructor that expects the list of collective
    variables and provides the necessary methods for initializing and executing
    the biasing during a simulation. Inheriting classes are expected to enhance
    or overwrite its methods as needed.
    """

    __special_args__ = set()
    snapshot_flags = set()

    def __init__(self, cvs, **kwargs):
        self.cvs = cvs
        self.cv = build(*cvs, differentiate=kwargs.get("cv_grad", True))
        self.requires_box_unwrapping = reduce(
            or_, (cv.requires_box_unwrapping for cv in cvs), False
        )
        self.kwargs = kwargs

    def __getstate__(self):
        return default_getstate(self)

    def __setstate__(self, state):
        default_setstate(self, state)

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        """
        Returns the snapshot, and two functions, ``initialize`` and ``update``.

        The ``initialize`` function is intended to allocate any run-time
        information required by the ``update`` function. The ``update``
        function is called after each integration step, that is, after each
        call to the wrapped context's ``run`` method.
        """


class GriddedSamplingMethod(SamplingMethod):
    """Base class for sampling methods that use grids."""

    __special_args__ = {"grid"}

    def __init__(self, cvs, grid, **kwargs):
        check_dims(cvs, grid)
        super().__init__(cvs, **kwargs)
        self.grid = grid
        self.restraints = canonicalize(kwargs.get("restraints", None), cvs)

    def __getstate__(self):
        return (get_info(self.grid), *default_getstate(self))

    def __setstate__(self, state):
        grid_args, args, kwargs = state
        args["grid"] = build_grid(*grid_args)
        default_setstate(self, (args, kwargs))

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


class NNSamplingMethod(GriddedSamplingMethod):
    """Base class for sampling methods that use neural networks."""

    def __init__(self, cvs, grid, topology, **kwargs):
        super().__init__(cvs, grid, **kwargs)
        self.topology = topology

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


#  Main functions
#  ==============


@dispatch.abstract
def run(method_or_result, context_generator, timesteps, **kwargs):
    """
    Runs a single or multiple replicas of a simulation with the specified sampling method.

    **Note**: Many specializations for this method are provided.
    """


@dispatch
def run(  # noqa: F811 # pylint: disable=C0116,E0102
    method: SamplingMethod,
    context_generator: Callable,
    timesteps: Union[int, float],
    callback: Optional[Callable] = None,
    context_args: dict = {},
    post_run_action: Optional[Callable] = None,
    config: ReplicasConfiguration = ReplicasConfiguration(),
    **kwargs,
):
    # """
    # Parameters
    # ----------

    # method_or_result: Union[SamplingMethod, Result]

    # context_generator: Callable
    #     User defined function that sets up a simulation context with the backend.
    #     Must return an instance of hoomd.context.SimulationContext for HOOMD-blue
    #     and openmm.Simulation for OpenMM. The function gets context_args
    #     unpacked for additional user arguments.

    # timesteps: int
    #     Number of time steps the simulation is running.

    # callback: Optional[Callable] = None
    #     Allows for user defined actions into the simulation workflow of the method.
    #     kwargs gets passed to the backend run function.

    # context_args: dict = {}
    #     Arguments to pass down to context_generator to setup the simulation context.

    # post_run_action: Optional[Callable] = None
    #     Callable function that enables actions after the run execution of PySAGES.
    #     Actions are executed inside the generated context. Example uses for this
    #     include writing a final configuration file. This function gets context_args
    #     unpacked just like context_generator.

    # config: ReplicasConfiguration = ReplicasConfiguration()
    #     Specifies the number of replicas of the simulation to generate.
    #     It also contains an executor which will manage different process
    #     or threads in case the multiple simulation are to be run in parallel.
    #     Defaults to ReplicasConfiguration(1, SerialExecutor()),
    #     which means only one simulation is run.
    # """
    timesteps = int(timesteps)

    def submit_work(executor, method, callback):
        return executor.submit(
            _run_replica,
            method,
            context_generator,
            timesteps,
            context_args,
            callback,
            post_run_action,
            **kwargs,
        )

    with config.executor as ex:
        futures = [submit_work(ex, method, callback) for _ in range(config.copies)]
    results = [future.result() for future in futures]
    states = [r.states for r in results]
    snapshots = [r.snapshots for r in results]
    callbacks = None if callback is None else [r.callbacks for r in results]

    return Result(method, states, callbacks, snapshots)


@dispatch
def run(  # noqa: F811 # pylint: disable=C0116,E0102
    sampling_context: SamplingContext,
    timesteps: Union[int, float],
    **kwargs,
):
    # """
    # Alternative interface for running a simulation from a SamplingContext.

    # Parameters
    # ----------
    # sampling_context: SamplingContext
    #     Instance of the simulation context of one of the supported backends
    #     wrapped as a SamplingContext.
    # timesteps: int
    #     Number of time steps the simulation is running.
    # kwargs: dict
    #     These gets passed to the backend run function.

    # Notes
    # -----
    # This interface supports only single replica runs.
    # """
    method = sampling_context.method
    method_type = type(method)
    if has_custom_run(method_type):
        raise RuntimeError(
            f"Method {method_type} is not compatible with the SamplingContext interface "
            "use pysages.run(method, context_generator, timesteps) instead."
        )

    timesteps = int(timesteps)
    sampler = sampling_context.sampler
    callback = None if sampler.callback is None else [sampler.callback]

    with sampling_context:
        sampling_context.run(timesteps, **kwargs)

    return Result(method, [sampler.state], callback, [sampler.take_snapshot()])


@dispatch
def run(  # noqa: F811 # pylint: disable=C0116,E0102
    result: Result,
    context_generator: Callable,
    timesteps: Union[int, float],
    context_args: dict = {},
    post_run_action: Optional[Callable] = None,
    config: ReplicasConfiguration = ReplicasConfiguration(),
    **kwargs,
):
    # """
    # Restart a simulation from a previously stored result.
    # """
    method = result.method
    timesteps = int(timesteps)
    callbacks_ = result.callbacks
    callbacks = [None] * len(result.states) if callbacks_ is None else callbacks_

    def submit_work(executor, result):
        return executor.submit(
            _run_replica,
            result,
            context_generator,
            timesteps,
            context_args,
            post_run_action,
            **kwargs,
        )

    with config.executor as ex:
        result_args = zip(result.states, callbacks, result.snapshots)
        futures = [submit_work(ex, ReplicaResult(method, *args)) for args in result_args]

    results = [future.result() for future in futures]
    states = [r.states for r in results]
    snapshots = [r.snapshots for r in results]
    callbacks = None if callbacks_ is None else [r.callbacks for r in results]

    return Result(method, states, callbacks, snapshots)


def _run_replica(method, *args, **kwargs):
    # Trampoline method to enable multiple replicas to be run with mpi4py.
    run = dispatch_table(dispatch)["_run"]
    return run(method, *args, **kwargs)


@dispatch
def _run(  # noqa: F811 # pylint: disable=C0116,E0102
    method: SamplingMethod,
    context_generator: Callable,
    timesteps: Union[int, float],
    context_args: dict = {},
    callback: Optional[Callable] = None,
    post_run_action: Optional[Callable] = None,
    **kwargs,
):
    """
    Base implementation for running a single simulation with the specified `SamplingMethod`.

    Parameters
    ----------

    method: SamplingMethod

    context_generator: Callable
        User defined function that sets up a simulation context with the backend.
        Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
        and `openmm.Simulation` for OpenMM. The function gets `context_args`
        unpacked for additional user arguments.

    timesteps: int
        Number of time steps the simulation is running.

    context_args: dict = {}
        Arguments to pass down to `context_generator` to setup the simulation context.

    callback: Optional[Callable] = None
        Allows for user defined actions into the simulation workflow of the method.
        `kwargs` gets passed to the backend `run` function. Default value is `None`.

    post_run_action: Optional[Callable] = None
        Callable function that enables actions after the run execution of PySAGES.
        Actions are executed inside the generated context. Example uses for this
        include writing a final configuration file. This function gets `context_args`
        unpacked just like `context_generator`.

    **Note**: All arguments must be pickable.
    """
    timesteps = int(timesteps)

    sampling_context = SamplingContext(method, context_generator, callback, context_args)
    context_args["context"] = sampling_context.context
    sampler = sampling_context.sampler

    with sampling_context:
        sampling_context.run(timesteps, **kwargs)
        if post_run_action:
            post_run_action(**context_args)

    return ReplicaResult(method, sampler.state, callback, sampler.take_snapshot())


@dispatch
def _run(  # noqa: F811 # pylint: disable=C0116,E0102
    result: ReplicaResult,
    context_generator: Callable,
    timesteps: Union[int, float],
    context_args: dict = {},
    post_run_action: Optional[Callable] = None,
    **kwargs,
):
    """
    Base implementation for running a single simulation from a previously stored `Result`.
    """
    timesteps = int(timesteps)
    method = result.method
    callback = result.callbacks

    sampling_context = SamplingContext(method, context_generator, callback, context_args)
    context_args["context"] = sampling_context.context
    sampler = sampling_context.sampler
    prev_snapshot = result.snapshots
    if device_platform(sampler.state.xi) == "cpu":
        prev_snapshot = copy(prev_snapshot, ToCPU())
    sampler.restore(prev_snapshot)
    sampler.state = result.states

    with sampling_context:
        sampling_context.run(timesteps, **kwargs)
        if post_run_action:
            post_run_action(**context_args)

    return ReplicaResult(method, sampler.state, callback, sampler.take_snapshot())


@dispatch.abstract
def analyze(result: Result):
    pass


#  Utils
#  =====


def default_getstate(method: SamplingMethod):
    init_args = set(getfullargspec(method.__init__).args[1:]) - method.__special_args__
    return {key: method.__dict__[key] for key in init_args}, method.kwargs


def default_setstate(method, state):
    args, kwargs = state
    method.__init__(**args, **kwargs)


@dispatch
def check_dims(cvs, grid: Grid):
    if len(cvs) != grid.shape.size:
        raise ValueError("Grid and Collective Variable dimensions must match.")


@dispatch
def check_dims(cvs, grid: type(None)):  # noqa: F811 # pylint: disable=C0116,E0102
    pass


@dispatch
def get_method(method: SamplingMethod):
    return method


@dispatch
def get_method(result: Result):  # noqa: F811 # pylint: disable=C0116,E0102
    return result.method


def has_custom_run(method: type):
    """
    Determine if ``method`` has a specialized ``run`` implementation.
    """
    return has_method(dispatch_table(dispatch)["run"], method, 0)


def generalize(concrete_update, helpers, jit_compile=True):
    if jit_compile:
        _jit = jit
    else:
        _jit = identity

    _update = _jit(concrete_update)

    def update(snapshot, state):
        return _update(state, helpers.query(snapshot))

    return _jit(update)
