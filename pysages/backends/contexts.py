# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
This module defines "Context" classes for backends that do not provide a dedicated Python
class to hold the simulation data.
"""

import weakref
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from xml.etree import ElementTree as et

from pysages.typing import (
    Any,
    Callable,
    Iterable,
    JaxArray,
    NamedTuple,
    Optional,
    Union,
)
from pysages.utils import dispatch, is_file, splitlines

JaxMDState = Any
QboxInstance = Any
XMLElement = et.Element


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


@dataclass(frozen=True)
class QboxContextGenerator:
    """
    Provides an interface for setting up Qbox-backed simulations.

    Arguments
    ---------

    launch_command: str
        Specifies the command that will be used to run Qbox in interactive mode,
        e.g. `qb` or `mpirun -n 4 qb`.

    script: str
        File or multile string with the Qbox input script.

    nitscf: Optional[int]
        Same as Qbox's `run` command parameter. The maximum number of self-consistent
        iterations.

    nite: Optional[int]
        Same as Qbox's `run` command parameter. The number of electronic iterations
        performed between updates of the charge density.

    logfile: Union[Path, str]
        Name for the output file. It must not exist on the working directory.
        Defaults to `qb.r`.
    """

    # NOTE: we leave `niter` as non-configurable for now.
    # niter: int
    #     Same as Qbox's `run` command parameter. The number of steps during which atomic
    #     positions are updated. Defaults to 1.

    launch_command: str
    script: str
    nitscf: Optional[int] = None
    nite: Optional[int] = None
    logfile: Union[Path, str] = Path("qb.r")

    def __call__(self, **kwargs):
        if is_file(self.logfile):
            msg = f"Rename or delete {self.logfile}, or choose a different log file name"
            raise FileExistsError(msg)

        return QboxContext(
            self.launch_command, self.script, self.logfile, 1, self.nitscf, self.nite
        )


@dataclass(frozen=True)
class QboxContext:
    instance: QboxInstance
    niter: int
    nitscf: Optional[int]
    nite: Optional[int]
    species_masses: dict
    initial_state: XMLElement
    state: XMLElement

    @dispatch
    def __init__(
        self, launch_command: str, script: str, logfile: Union[Path, str], niter, nitscf, nite
    ):
        pexpect = import_module("pexpect.popen_spawn")

        def finalize(qb):
            if not qb.flag_eof:
                qb.sendline("quit")
                qb.expect(pexpect.EOF)

        qb = pexpect.PopenSpawn(launch_command)
        weakref.finalize(qb, lambda: finalize(qb))
        qb.logfile_read = open(logfile, "wb")
        i = qb.expect([r"\[qbox\] ", pexpect.EOF])

        if i == 1:  # EOF was written to the log file
            preamble = (
                "The command:\n\n  "
                f"{launch_command}\n\n"
                "for running Qbox failed, it returned the following:\n\n"
            )
            raise ChildProcessError(preamble + qb.before.decode())

        super().__setattr__("instance", qb)
        super().__setattr__("niter", niter)
        super().__setattr__("nitscf", "" if nitscf is None else nitscf)
        super().__setattr__("nite", "" if nite is None else nite)

        initial_state = qb.before
        state = self.process_input(script)  # sets `self.state`

        if self.state.find("error") is not None:
            try:
                qb.expect(pexpect.EOF, timeout=3)
            finally:
                raise ChildProcessError("Qbox encountered the following error:\n" + state.decode())

        initial_state += state + b"\n</fpmd:simulation>"
        super().__setattr__("initial_state", et.fromstring(initial_state))

        k = 1822.888486  # to convert amu to atomic units
        species = self.initial_state.iter("species")
        species_masses = {s.attrib["name"]: k * float(s.find("mass").text) for s in species}
        super().__setattr__("species_masses", species_masses)

    def process_input(self, entries: Union[str, Iterable[str]], target=r"\[qbox\] ", timeout=None):
        qb = self.instance
        state = b""
        for entry in splitlines(entries):
            qb.sendline(entry)
            qb.expect(target, timeout=timeout)
            state += qb.before
        # We add tags to ensure that the state corresponds to a valid xml section
        super().__setattr__("state", et.fromstring(b"<root>\n" + state + b"\n</root>"))
        return state
