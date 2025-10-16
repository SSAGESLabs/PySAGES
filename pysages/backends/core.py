# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from importlib import import_module

from pysages.backends.contexts import JaxMDContext
from pysages.typing import Callable, Optional


class SamplingContext:
    """
    PySAGES simulation context. Manages access to the backend-dependent simulation context.
    """

    def __init__(
        self,
        sampling_method,
        context_generator: Callable,
        callback: Optional[Callable] = None,
        context_args: dict = {},
        **kwargs,
    ):
        """
        Automatically identifies the backend and binds the sampling method to
        the simulation context.
        """
        self._backend_name = None
        context = context_generator(**context_args)
        module_name = type(context).__module__

        if module_name.startswith("ase.md"):
            self._backend_name = "ase"
        elif module_name.startswith("hoomd"):
            self._backend_name = "hoomd"
        elif isinstance(context, JaxMDContext):
            self._backend_name = "jax-md"
        elif module_name.startswith("lammps"):
            self._backend_name = "lammps"
        elif module_name.startswith("simtk.openmm") or module_name.startswith("openmm"):
            self._backend_name = "openmm"

        if self._backend_name is None:
            backends = ", ".join(supported_backends())
            raise ValueError(f"Invalid backend {module_name}: supported options are ({backends})")

        self.context = context
        self.method = sampling_method
        self.run = None

        backend = import_module("." + self._backend_name, package="pysages.backends")
        self.sampler = backend.bind(self, callback, **kwargs)

        # `self.run` *must* be set by the backend bind function.
        assert self.run is not None

    @property
    def backend_name(self):
        return self._backend_name

    def __enter__(self):
        """
        Trampoline 'with statements' to the wrapped context when the backend supports it.
        """
        if hasattr(self.context, "__enter__"):
            return self.context.__enter__()
        return self.context

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Trampoline 'with statements' to the wrapped context when the backend supports it.
        """
        if hasattr(self.context, "__exit__"):
            self.context.__exit__(exc_type, exc_value, exc_traceback)


def supported_backends():
    return ("ase", "hoomd", "jax-md", "lammps", "openmm")
