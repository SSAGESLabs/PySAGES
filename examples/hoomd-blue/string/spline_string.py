#!/usr/bin/env python3
import argparse
import importlib
import sys

import hoomd
import hoomd.dlext
import hoomd.md as md
import matplotlib.pyplot as plt
import numpy as np

import pysages
from pysages.colvars import Component
from pysages.methods import SerialExecutor, SplineString

params = {"A": 0.5, "w": 0.2, "p": 2}


def generate_context(**kwargs):
    if kwargs.get("mpi_enabled") and False:
        MPI = importlib.import_module("mpi4py.MPI")
        init_kwargs = {"mpi_comm": MPI.COMM_SELF}
    else:
        init_kwargs = {}
    hoomd.context.initialize("--single-mpi", **init_kwargs)
    context = hoomd.context.SimulationContext()

    with context:
        print(f"Operating replica {kwargs.get('replica_num')}")
        hoomd.init.read_gsd("start.gsd")

        md.integrate.nve(group=hoomd.group.all())
        md.integrate.mode_standard(dt=0.01)

        nl = md.nlist.cell()
        dpd = md.pair.dpd(r_cut=1, nlist=nl, seed=42, kT=1.0)
        dpd.pair_coeff.set("A", "A", A=5.0, gamma=1.0)
        dpd.pair_coeff.set("A", "B", A=5.0, gamma=1.0)
        dpd.pair_coeff.set("B", "B", A=5.0, gamma=1.0)

        periodic = md.external.periodic()
        periodic.force_coeff.set("A", A=params["A"], i=0, w=params["w"], p=params["p"])
        periodic.force_coeff.set("B", A=0.0, i=0, w=0.02, p=1)

    return context


def external_field(r, A, p, w):
    return A * np.tanh(1 / (2 * np.pi * p * w) * np.cos(p * r))


def get_args(argv):
    available_args = [
        ("k-spring", "k", float, 50, "Spring constant for each replica"),
        ("replicas", "N", int, 25, "Number of replicas along the path"),
        ("time-steps", "t", int, 1e5, "Number of simulation steps for each replica"),
        ("log-period", "l", int, 50, "Frequency of logging the CVs into each histogram"),
        ("log-delay", "d", int, 5e3, "Number of timesteps to discard before logging"),
        ("start-path", "s", float, -1.5, "Start point of the path"),
        ("end-path", "e", float, 1.5, "Start point of the path"),
        ("string-steps", "p", int, 15, "Iteration of the string algorithm"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run string method.")
    for (name, short, T, val, doc) in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    parser.add_argument("--mpi", action="store_true", help="Use MPI executor")
    args = parser.parse_args(argv)
    return args


def post_run_action(**kwargs):
    hoomd.option.set_notice_level(0)
    hoomd.dump.gsd(
        filename=f"final_{kwargs.get('stringstep')}_{kwargs.get('replica_num')}.gsd",
        overwrite=True,
        period=None,
        group=hoomd.group.all(),
    )
    hoomd.option.set_notice_level(2)


def get_executor(args):
    if args.mpi:
        futures = importlib.import_module("mpi4py.futures")
        return futures.MPIPoolExecutor()
    return SerialExecutor()


def plot_energy(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Free energy $[\\epsilon]$")
    centers = np.asarray(np.asarray(result["path"])[:, 0])
    free_energy = np.asarray(result["free_energy"])
    offset = np.min(free_energy)
    ax.plot(centers, free_energy - offset, "o", color="teal")

    x = np.linspace(-3, 3, 50)
    data = external_field(x, **params)
    offset = np.min(data)
    ax.plot(x, data - offset, label="test")

    fig.savefig("energy.pdf")


def main(argv):
    args = get_args(argv)

    cvs = [Component([0], 0), Component([0], 1), Component([0], 2)]

    centers = [[c, -1, 1] for c in np.linspace(args.start_path, args.end_path, args.replicas)]
    method = SplineString(cvs, args.k_spring, centers, 1e-2, args.log_period, args.log_delay)

    context_args = {"mpi_enabled": args.mpi}

    raw_result = pysages.run(
        method,
        generate_context,
        args.time_steps,
        args.string_steps,
        context_args=context_args,
        post_run_action=post_run_action,
        executor=get_executor(args),
    )
    result = pysages.analyze(raw_result)
    print(np.asarray(result["path_history"]))
    print(result["path"])
    print(result["point_convergence"])
    plot_energy(result)


if __name__ == "__main__":
    main(sys.argv[1:])
