#!/usr/bin/env python3

"""
Example SpectralABF simulation with pysages and lammps.

For a list of possible options for running the script pass `-h` as argument from the
command line, or call `get_args(["-h"])` if the module was loaded interactively.
"""

# %%
import argparse
import sys

import numpy
from lammps import lammps

import pysages
from pysages.colvars import DihedralAngle
from pysages.methods import HistogramLogger, SpectralABF


# %%
def generate_context(args="", script="adp.lmp", store_freq=1):
    """
    Returns a lammps simulation defined by the contents of `script` using `args` as
    initialization arguments.
    """
    context = lammps(cmdargs=args.split())
    context.file(script)
    # Allow for the retrieval of the wrapped positions
    context.command(f"dump 4a all custom {store_freq} dump.myforce id type x y z")
    return context


def get_args(argv):
    """Process the command-line arguments to this script."""

    available_args = [
        ("time-steps", "t", int, 2e6, "Number of simulation steps"),
        ("kokkos", "k", bool, True, "Whether to use Kokkos acceleration"),
        ("log-steps", "l", int, 2e3, "Number of simulation steps for logging"),
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
    """Example SpectralABF simulation with pysages and lammps."""
    args = get_args(argv)

    context_args = {"store_freq": args.log_steps}
    if args.kokkos:
        # Passed to the lammps constructor as `cmdargs` when running the script
        # with the --kokkos (or -k) option
        context_args["args"] = "-k on g 1 -sf kk -pk kokkos newton on neigh half"
    #        context_args["args"] = "-k on -sf kk -pk kokkos newton on neigh half"

    # Setting the collective variable, method, and running the simulation
    cvs = [DihedralAngle([4, 6, 8, 14]), DihedralAngle([6, 8, 14, 16])]
    grid = pysages.Grid(
        lower=(-numpy.pi, -numpy.pi),
        upper=(numpy.pi, numpy.pi),
        shape=(32, 32),
        periodic=True,
    )
    method = SpectralABF(cvs, grid)
    callback = HistogramLogger(args.log_steps)
    raw_result = pysages.run(
        method,
        generate_context,
        args.time_steps,
        callback=callback,
        context_args=context_args,
    )
    # Post-run analysis
    result = pysages.analyze(raw_result)
    mesh = result["mesh"]
    fes_fn = result["fes_fn"]
    A = fes_fn(mesh)
    hist = result["histogram"]
    A = A.reshape(32, 32)
    numpy.savetxt("FES.dat", numpy.hstack([mesh, A.reshape(-1, 1)]))
    numpy.savetxt("hist.dat", numpy.hstack([mesh, hist.reshape(-1, 1)]))
    bins = 50
    histo, xedges, yedges = numpy.histogram2d(
        callback.data[:, 0],
        callback.data[:, 1],
        bins=bins,
        range=[[-numpy.pi, numpy.pi], [-numpy.pi, numpy.pi]],
    )
    xedges = (xedges[1:] + xedges[:-1]) / 2
    yedges = (yedges[1:] + yedges[:-1]) / 2
    mesh = numpy.dstack(numpy.meshgrid(xedges, yedges)).reshape(-1, 2)
    numpy.savetxt("hist-from-logger.dat", numpy.hstack([mesh, histo.reshape(-1, 1)]))


if __name__ == "__main__":
    main(sys.argv[1:])
