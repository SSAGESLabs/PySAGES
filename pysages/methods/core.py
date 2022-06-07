# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC, abstractmethod
from functools import reduce
from operator import or_
from typing import Callable, Optional, Union

from jax import grad, jit
from plum import parametric

from pysages.backends import ContextWrapper
from pysages.colvars.core import build
from pysages.utils import dispatch, identity


#  Base Classes
#  ============


class SamplingMethod(ABC):
    """
    Abstract base class for all sampling methods.

    Defines the constructor that expects the collective variables,
    the build method to initialize the GPU execution for the biasing,
    and the run method that executes the simulation run.
    All these are intended to be enhanced/overwritten by inheriting classes.
    """

    snapshot_flags = set()

    def __init__(self, cvs, *args, **kwargs):
        self.cvs = cvs
        self.cv = build(*cvs, grad=kwargs.get("cv_grad", grad))
        self.requires_box_unwrapping = reduce(
            or_, (cv.requires_box_unwrapping for cv in cvs), False
        )
        self.args = args
        self.kwargs = kwargs
        self.context = []

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        """
        Returns the snapshot, and two functions: `initialize` and `update`.
        `initialize` is intended to allocate any runtime information required
        by `update`, while `update` is intended to be called after each call to
        the wrapped context's `run` method.
        """
        pass


class GriddedSamplingMethod(SamplingMethod):
    def __init__(self, cvs, grid, *args, **kwargs):
        check_dims(cvs, grid)
        super().__init__(cvs, *args, **kwargs)
        self.grid = grid

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


class NNSamplingMethod(GriddedSamplingMethod):
    def __init__(self, cvs, grid, topology, *args, **kwargs):
        super().__init__(cvs, grid, *args, **kwargs)
        self.topology = topology

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


@parametric
class Result:
    @classmethod
    def __infer_type_parameter__(self, method, *args):
        return type(method)

    @dispatch
    def __init__(self, method: SamplingMethod, states, callbacks=None):
        self.method = method
        self.states = states
        self.callbacks = callbacks


#  Main functions
#  ==============


@dispatch
def run(
    method: SamplingMethod,
    context_generator: Callable,
    timesteps: Union[int, float],
    callback: Optional[Callable] = None,
    context_args: Optional[dict] = None,
    **kwargs
):
    """
    Base implementation for running a single simulation with the specified `SamplingMethod`.

    Arguments
    ---------
    method: SamplingMethod

    context_generator: Callable
        User defined function that sets up a simulation context with the backend.
        Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
        and `openmm.Simulation` for OpenMM. The function gets `context_args`
        unpacked for additional user arguments.

    timesteps: int
        Number of timesteps the simulation is running.

    callback: Optional[Callable] = None
        Allows for user defined actions into the simulation workflow of the method.
        `kwargs` gets passed to the backend `run` function.

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.
    """
    timesteps = int(timesteps)
    context_args = {} if context_args is None else context_args

    context = context_generator(**context_args)
    wrapped_context = ContextWrapper(context, method, callback)

    with wrapped_context:
        wrapped_context.run(timesteps, **kwargs)

    return wrapped_context.sampler.state


@dispatch.abstract
def analyze(result: Result):
    pass


#  Utils
#  =====


def check_dims(cvs, grid):
    if len(cvs) != grid.shape.size:
        raise ValueError("Grid and Collective Variable dimensions must match.")


def generalize(concrete_update, helpers, jit_compile=True):
    if jit_compile:
        _jit = jit
    else:
        _jit = identity

    _update = _jit(concrete_update)

    def update(snapshot, state):
        return _update(state, helpers.query(snapshot))

    return _jit(update)
