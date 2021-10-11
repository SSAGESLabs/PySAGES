# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# pylint: disable=unused-import,relative-beyond-top-level
# flake8: noqa F401
"""Combine 'runner' functions for PySAGES.
   Runner functions take a user function to generate a simulation context and a method to run a specific integration scheme for this context and method.
"""
from typing import Callable
from pysages.backends import bind


def run_simple(context_generator: Callable, method, timesteps:int, callback: Callable, context_args=dict(), **kwargs):
    """
    Run a simple simulation with 1 replica and just one context.
    context_generator: user function that generates a simulation context for the backend.
                       The function gets kwargs passed for additional user args.
    method: PySAGES method to be integrated in the simulation run.
    timesteps: number of timesteps to run the simulation.
    callback: Callback to integrate user defined actions into the simulation workflow of the method
    context_args: are arguments in a dictionary, that gets unpack and passed to the user context generator function.

    kwargs gets passed to the backend run function for additional user arguments to be passed down.
    """

    _run_simple_hoomd(context_generator, method, timesteps, callback, context_args, **kwargs)


def _run_simple_hoomd(context_generator: Callable, method, timesteps:int, callback: Callable, context_args, **kwargs):
    import hoomd
    with hoomd.context.SimulationContext() as context:
        context_generator(**context_args)
        bind(context, method, callback)
        hoomd.run(timesteps, **kwargs)


def _run_simple_openmm(context_generator: Callable, method, timesteps:int, callback: Callable, context_args, **kwargs):
    pass
