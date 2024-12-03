# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
This module defines "Context" classes for backends that do not provide a dedicated Python
class to hold the simulation data.
"""

from importlib import import_module

from pysages.typing import Any, Callable, JaxArray, NamedTuple, Optional
from pysages.utils import is_file

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


class QboxContextGenerator:
    """
    Provides an interface for setting up Qbox-backed simulations.

    Arguments
    ---------
    launch_command: str
        Specifies the command that will be used to run Qbox in interactive mode,
        e.g. `qb` or `mpirun -n 4 qb`.

    input_script: str
        Path to the Qbox input script.

    output_filename: Union[Path, str]
        Name for the output file. It must not exist on the working directory.
        Defaults to `qb.r`.
    """

    def __init__(self, launch_command, input_script, output_filename="qb.r"):
        self.cmd = launch_command
        self.script = input_script
        self.logfile = output_filename

    def __call__(self, **kwargs):
        if not is_file(self.script):
            raise FileNotFoundError(f"Unable to find or open {self.script}")

        if is_file(self.logfile):
            msg = f"Delete {self.logfile} or choose a different output file name"
            raise FileExistsError(msg)

        pexpect = import_module("pexpect")

        qb = pexpect.spawn(self.cmd)
        qb.logfile_read = open(self.logfile, "wb")
        qb.expect(r"\[qbox\] ")

        qb.sendline(self.script)
        qb.expect(r"\[qbox\] ")

        return qb
