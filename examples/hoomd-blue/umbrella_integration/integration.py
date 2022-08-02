#!/usr/bin/env python3
import sys
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt

import hoomd
import hoomd.md as md
import hoomd.dlext

import pysages
from pysages.colvars import Component
from pysages.methods import HarmonicBias, UmbrellaIntegration, SerialExecutor


params = {"A": 0.5, "w": 0.2, "p": 2}


def generate_context(**kwargs):
    if kwargs.get("mpi_enabled"):
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


def plot_hist(result, bins=50):
    fig, ax = plt.subplots(2, 2)

    # ax.set_xlabel("CV")
    # ax.set_ylabel("p(CV)")

    counter = 0
    hist_per = len(result["centers"]) // 4 + 1
    for x in range(2):
        for y in range(2):
            for i in range(hist_per):
                if counter + i < len(result["centers"]):
                    center = np.asarray(result["centers"][counter + i])
                    histo, edges = result["histograms"][counter + i].get_histograms(bins=bins)
                    edges = np.asarray(edges)[0]
                    edges = (edges[1:] + edges[:-1]) / 2
                    ax[x, y].plot(edges, histo, label=f"center {center}")
                    ax[x, y].legend(loc="best", fontsize="xx-small")
                    ax[x, y].set_yscale("log")
            counter += hist_per
    while counter < len(result["centers"]):
        center = np.asarray(result["centers"][counter])
        histo, edges = result["histograms"][counter].get_histograms(bins=bins)
        edges = np.asarray(edges)[0]
        edges = (edges[1:] + edges[:-1]) / 2
        ax[1, 1].plot(edges, histo, label=f"center {center}")
        counter += 1

    fig.savefig("hist.pdf")


def external_field(r, A, p, w):
    return A * np.tanh(1 / (2 * np.pi * p * w) * np.cos(p * r))


def plot_energy(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Free energy $[\\epsilon]$")
    centers = np.asarray(result["centers"])
    free_energy = np.asarray(result["free_energy"])
    offset = np.min(free_energy)
    ax.plot(centers, free_energy - offset, color="teal")

    x = np.linspace(-3, 3, 50)
    data = external_field(x, **params)
    offset = np.min(data)
    ax.plot(x, data - offset, label="test")

    fig.savefig("energy.pdf")


def get_args(argv):
    available_args = [
        ("k-spring", "k", float, 50, "Spring constant for each replica"),
        ("replicas", "N", int, 25, "Number of replicas along the path"),
        ("start-path", "s", float, -1.5, "Start point of the path"),
        ("end-path", "e", float, 1.5, "Start point of the path"),
        ("time-steps", "t", int, 1e5, "Number of simulation steps for each replica"),
        ("log-period", "l", int, 50, "Frequency of logging the CVs into each histogram"),
        ("log-delay", "d", int, 0, "Number of timesteps to discard before logging"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run umbrella integration")
    for (name, short, T, val, doc) in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    parser.add_argument("--mpi", action="store_true", help="Use MPI executor")
    args = parser.parse_args(argv)
    return args


def post_run_action(**kwargs):
    hoomd.dump.gsd(
        filename=f"final_{kwargs.get('replica_num')}.gsd",
        overwrite=True,
        period=None,
        group=hoomd.group.all(),
    )


def get_executor(args):
    if args.mpi:
        futures = importlib.import_module("mpi4py.futures")
        return futures.MPIPoolExecutor()
    return SerialExecutor()


def main(argv):
    args = get_args(argv)

    cvs = [Component([0], 0)]

    centers = list(np.linspace(args.start_path, args.end_path, args.replicas))
    biasers = [HarmonicBias(cvs, args.k_spring, c) for c in centers]
    method = UmbrellaIntegration(biasers, args.log_period, args.log_delay)

    context_args = {"mpi_enabled": args.mpi}

    raw_result = pysages.run(
        method,
        generate_context,
        args.time_steps,
        context_args=context_args,
        post_run_action=post_run_action,
        executor=get_executor(args),
    )
    result = pysages.analyze(raw_result)

    plot_energy(result)
    plot_hist(result)


if __name__ == "__main__":
    main(sys.argv[1:])
