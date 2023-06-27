# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from importlib import import_module
from typing import Any, Callable, NamedTuple, Optional

from pysages.utils import Float, JaxArray

JaxMDState = Any


class JaxMDContextState(NamedTuple):
    """
    Provides an interface for the data structure returned by `JaxMDContext.init_fn` and
    expected as the single argument of `JaxMDContext.step_fn`.

    Arguments
    ---------
    state: JaxMDState
        Holds the particle information and corresponds to the internal state of
        `jax_md.simulate` methods.

    extras: Optional[dict]
        Additional arguments required by `JaxMDContext.step_fn`, these might include for
        instance, the neighbor list or the time step.
    """

    state: JaxMDState
    extras: Optional[dict]


class JaxMDContext(NamedTuple):
    """
    Provides an interface for the data structure expects from `generate_context` for
    `jax_md`-backed simulations.

    Arguments
    ---------
    init_fn: Callable[..., JaxMDContextState]
        Initilizes the `jax_md` state. Generally, this will be the `init_fn` of any
        of the simulation routines in `jax_md` (or wrappers around these).

    step_fn: Callable[..., JaxMDContextState]
        Takes a state and advances a `jax_md` simulation by one step. Generally, this
        will be the `apply_fn` of any of the simulation routines in `jax_md` (or wrappers
        around these).

    box: JaxArray
        Affine transformation from a unit hypercube to the simulation box.

    dt: Float
        Step size of the simulation.
    """

    init_fn: Callable[..., JaxMDContextState]
    step_fn: Callable[..., JaxMDContextState]
    box: JaxArray
    dt: Float


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

        if self._backend_name is not None:
            self._backend = import_module("." + self._backend_name, package="pysages.backends")
        else:
            backends = ", ".join(supported_backends())
            raise ValueError(
                f"Invalid backend {self._backend_name}: supported options are ({backends})"
            )

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
        if hasattr(self.context, "__enter__"):
            return self.context.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Trampoline 'with statements' to the wrapped context when the backend supports it.
        """
        if hasattr(self.context, "__exit__"):
            return self.context.__exit__(exc_type, exc_value, exc_traceback)


def supported_backends():
    return ("ase", "hoomd", "jax-md", "lammps", "openmm")
