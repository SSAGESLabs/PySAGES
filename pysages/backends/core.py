# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from importlib import import_module
from typing import Callable

from jax import numpy as np

import jax
import warnings


# Set default floating point type for arrays in `jax` to `jax.f64`
jax.config.update("jax_enable_x64", True)


class ContextWrapper:
    """
    PySAGES simulation context. Manages access to the backend-dependent simulation context.
    """

    def __init__(self, context, sampling_method, callback: Callable = None, **kwargs):
        """
        Automatically identifies the backend and binds the sampling method to
        the simulation context.
        """
        self._backend_name = None
        module_name = type(context).__module__
        if module_name.startswith("hoomd"):
            self._backend_name = "hoomd"
        elif module_name.startswith("simtk.openmm") or module_name.startswith("openmm"):
            self._backend_name = "openmm"

        if self._backend_name is not None:
            self._backend = import_module('.' + self._backend_name, package="pysages.backends")
        else:
            backends = ", ".join(supported_backends())
            raise ValueError(f"Invalid backend: supported options are ({backends})")

        self.context = context
        self.view = None
        self.run = None
        self.sampler = self._backend.bind(self, sampling_method, callback, **kwargs)

        # `self.view` and `self.run` *must* be set by the backend bind function.
        assert self.view is not None
        assert self.run is not None

        self.synchronize = self.view.synchronize

    def get_backend_name(self):
        return self._backend_name

    def get_backend_module(self):
        return self._backend

    def __enter__(self):
        """
        Trampoline 'with statements' to the wrapped context when the backend supports it.
        """
        if self.get_backend_name() == "hoomd":
            return self.context.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Trampoline 'with statements' to the wrapped context when the backend supports it.
        """
        if self.get_backend_name() == "hoomd":
            return self.context.__exit__(exc_type, exc_value, exc_traceback)


def supported_backends():
    return ("hoomd", "openmm")
