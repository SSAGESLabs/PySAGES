# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import importlib
import jax
import warnings

from typing import Callable
from jax import numpy as np

from pysages.ssages.methods import SamplingMethod


# Set default floating point type for arrays in `jax` to `jax.f64`
jax.config.update("jax_enable_x64", True)

class ContextWrapper:
    """PySAGES simulation context. It manages access to the backend depend simulation context. And is initialized by one."""

    def __init__(self, context, sampling_method: SamplingMethod, callback: Callable = None, **kwargs):
        """ Initialize the context (automatically identifies the backend)."""
        if type(context).__module__.startswith("hoomd"):
            self._backend_name = "hoomd"
        elif type(context).__module__.startswith("simtk.openmm"):
            self._backend_name = "openmm"

        if self._backend_name in supported_backends():
            self._backend = importlib.import_module('.' + self._backend_name, package="pysages.backends")
        else:
            raise ValueError("Invalid backend: support options are {0}".format(str(supported_backends())))
        self.context = context

        # self.view and self.run *must* be set by the backend bind function.
        self.view = None
        self.run = None
        self._backend.bind(self, sampling_method, callback, **kwargs)
        if self.view is None:
            raise RuntimeError("Backend {0} did not set context.view. Implementation error of backend. Please report bug.".format(self.get_backend_name()))
        if self.run is None:
            raise RuntimeError("Backend {0} did not set context.run. Implementation error of backend. Please report bug.".format(self.get_backend_name()))

        self.synchronize = self.view.synchronize

    def get_backend_name(self):
        return self._backend_name

    def get_backend_module(self):
        return self._backend

    def __enter__(self):
        """Trampoline python 'with' functions for backends that support it."""
        if self.get_backend_name() == "hoomd":
            return self.context.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Trampoline python 'with' functions for backends that support it."""
        if self.get_backend_name() == "hoomd":
            return self.context.__exit__(exc_type, exc_value, exc_traceback)

def supported_backends():
    return ("hoomd", "openmm")



def bind(wrapped_context, sampling_method, callback: Callable = None, **kwargs):
    """Couples the sampling method to the simulation."""

    return wrapped_context.get.bind(wrapped_context, sampling_method, callback, **kwargs)
