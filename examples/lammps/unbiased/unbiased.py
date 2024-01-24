#!/usr/bin/env python3

"""
Example unbiased simulation with pysages and lammps.

For a list of possible options for running the script pass `-h` as argument from the
command line, or call `get_args(["-h"])` if the module was loaded interactively.
"""

# %%
import argparse
import sys

from lammps import lammps

import pysages
from pysages.backends import SamplingContext
from pysages.colvars import Component
from pysages.methods import Unbiased


# %%
def generate_context(args="", script="lj.lmp", store_freq=1):
    """
    Returns a lammps simulation defined by the contents of `script` using `args` as
    initialization arguments.
    """
    context = lammps(cmdargs=args.split())
    context.file(script)
    # Allow for the retrieval of the unwrapped positions
    context.command(f"fix unwrap all store/state {store_freq} xu yu zu")
    return context


def get_args(argv):
    """Process the command-line arguments to this script."""

    available_args = [
        ("time-steps", "t", int, 1e2, "Number of simulation steps"),
        ("kokkos", "k", bool, True, "Whether to use Kokkos acceleration"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run pysages with lammps")

    for name, short, T, val, doc in available_args:
        if T is bool:
            action = "store_" + str(val).lower()
            parser.add_argument("--" + name, "-" + short, action=action, help=doc)
        else:
            convert = (lambda x: int(float(x))) if T is int else T
            parser.add_argument("--" + name, "-" + short, type=convert, default=T(val), help=doc)

    return parser.parse_args(argv)


def main(argv):
    """Example simulation with pysages and lammps."""
    args = get_args(argv)

    context_args = {"store_freq": args.time_steps}
    if args.kokkos:
        # Passed to the lammps constructor as `cmdargs` when running the script
        # with the --kokkos (or -k) option
        context_args["args"] = "-k on g 1 -sf kk -pk kokkos newton on neigh half"

    # Setting the collective variable, method, and running the simulation
    cvs = [Component([0], i) for i in range(3)]
    method = Unbiased(cvs)
    sampling_context = SamplingContext(method, generate_context, context_args=context_args)
    result = pysages.run(sampling_context, args.time_steps)

    # Post-run analysis
    # -----------------
    context = sampling_context.context
    nlocal = sampling_context.sampler.view.local_particle_number()
    snapshot = result.snapshots[0]
    state = result.states[0]

    # Retrieve the pointer to the unwrapped positions,
    ptr = context.extract_fix("unwrap", 1, 2)
    # and make them available as a numpy ndarray
    positions = context.numpy.darray(ptr, nlocal, dim=3)
    # Get the map to sort the atoms since they can be reordered during the simulation
    ids = context.numpy.extract_atom("id").argsort()

    # The ids of the final snapshot in pysages and lammps should be the same
    assert (snapshot.ids == ids).all()
    # For our example, the last value of the CV should match
    # the unwrapped position of the zeroth atom
    assert (state.xi.flatten() == positions[ids[0]]).all()


if __name__ == "__main__":
    main(sys.argv[1:])
