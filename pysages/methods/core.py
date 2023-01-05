# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABCMeta, abstractmethod
from functools import reduce
from inspect import getfullargspec
from operator import or_
from sys import modules as sys_modules
from typing import Callable, Optional, Union

from jax import jit
from plum import parametric

from pysages.backends import SamplingContext
from pysages.colvars.core import build
from pysages.grids import Grid, build_grid, get_info
from pysages.methods.restraints import canonicalize
from pysages.methods.utils import ReplicasConfiguration, methods_dispatch
from pysages.utils import dispatch, identity

#  Base Classes
#  ============


@parametric
class Result:
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

    Defines the constructor that expects the collective variables,
    the build method to initialize the GPU execution for the biasing,
    and the run method that executes the simulation run.
    All these are intended to be enhanced/overwritten by inheriting classes.
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
        Returns the snapshot, and two functions: `initialize` and `update`.
        `initialize` is intended to allocate any run time information required
        by `update`, while `update` is intended to be called after each call to
        the wrapped context's `run` method.
        """
        pass


class GriddedSamplingMethod(SamplingMethod):
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
    def __init__(self, cvs, grid, topology, **kwargs):
        super().__init__(cvs, grid, **kwargs)
        self.topology = topology

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


#  Main functions
#  ==============


@dispatch
def run(
    method: SamplingMethod,
    context_generator: Callable,
    timesteps: Union[int, float],
    callback: Optional[Callable] = None,
    context_args: Optional[dict] = None,
    post_run_action: Optional[Callable] = None,
    config: ReplicasConfiguration = ReplicasConfiguration(),
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

    callback: Optional[Callable] = None
        Allows for user defined actions into the simulation workflow of the method.
        `kwargs` gets passed to the backend `run` function.

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.

    post_run_action: Optional[Callable] = None
        Callable function that enables actions after the run execution of PySAGES.
        Actions are executed inside the generated context. Example uses for this
        include writing a final configuration file. This function gets `context_args`
        unpacked just like `context_generator`.

    config: ReplicasConfiguration = ReplicasConfiguration()
        Specifies the number of replicas of the simulation to generate.
        It also contains an `executor` which will manage different process
        or threads in case the multiple simulation are to be run in parallel.
        Defaults to `ReplicasConfiguration(1, SerialExecutor())`,
        which means only one simulation is run.
    """
    timesteps = int(timesteps)
    context_args = {} if context_args is None else context_args

    def submit_work(executor, method, callback):
        return executor.submit(
            _run,
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
    """
    Alternative interface for running a simulation from a `SamplingContext`.

    Parameters
    ----------

    sampling_context: SamplingContext
        Instance of the simulation context of one of the supported backends
        wrapped as a `SamplingContext`.

    timesteps: int
        Number of time steps the simulation is running.

    kwargs: dict
        These gets passed to the backend `run` function.

    NOTE: This interface supports only single replica runs.
    """
    timesteps = int(timesteps)
    method = sampling_context.method
    sampler = sampling_context.sampler
    callback = None if sampler.callback is None else [sampler.callback]

    with sampling_context:
        sampling_context.run(timesteps, **kwargs)

    return Result(method, [sampler.state], callback, [sampler.take_snapshot()])


def _run(method, *args, **kwargs):
    run = methods_dispatch._functions["run"]
    return run(method, *args, **kwargs)


# We use `methods_dispatch` instead of the global dispatcher `dispatch` to
# separate the definitions for a single run vs multiple replica simulations.
@methods_dispatch
def run(  # noqa: F811 # pylint: disable=C0116,E0102
    method: SamplingMethod,
    context_generator: Callable,
    timesteps: Union[int, float],
    context_args: Optional[dict] = None,
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

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.

    callback: Optional[Callable] = None
        Allows for user defined actions into the simulation workflow of the method.
        `kwargs` gets passed to the backend `run` function. Default value is `None`.

    post_run_action: Optional[Callable] = None
        Callable function that enables actions after the run execution of PySAGES.
        Actions are executed inside the generated context. Example uses for this
        include writing a final configuration file. This function gets `context_args`
        unpacked just like `context_generator`.

    *Note*: All arguments must be pickable.
    """
    timesteps = int(timesteps)

    context_args = {} if context_args is None else context_args
    context = context_generator(**context_args)
    context_args["context"] = context
    sampling_context = SamplingContext(context, method, callback)
    sampler = sampling_context.sampler

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


def generalize(concrete_update, helpers, jit_compile=True):
    if jit_compile:
        _jit = jit
    else:
        _jit = identity

    _update = _jit(concrete_update)

    def update(snapshot, state):
        return _update(state, helpers.query(snapshot))

    return _jit(update)
