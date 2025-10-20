# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
This module defines "Context" classes for backends that do not provide a dedicated Python
class to hold the simulation data.
"""

from pysages.typing import Any, Callable, JaxArray, NamedTuple, Optional

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

    dt: float
        Step size of the simulation.
    """

    init_fn: Callable[..., JaxMDContextState]
    step_fn: Callable[..., JaxMDContextState]
    box: JaxArray
    dt: float
