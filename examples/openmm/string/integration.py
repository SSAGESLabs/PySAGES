#!/usr/bin/env python3
import argparse
import importlib
import os
import shutil
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pysages
from pysages.colvars import DihedralAngle
from pysages.methods import SerialExecutor, SplineString
from pysages.utils import try_import

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


def internal_openmm_prep(
    pdb_filename="adp-explicit.pdb",
    T=298.15 * unit.kelvin,
    dt=2.0 * unit.femtoseconds,
    cutoff_distance=1.0 * unit.nanometer,
):
    pdb = app.PDBFile(pdb_filename)
    topology = pdb.topology
    ff = app.ForceField("amber99sb.xml", "tip3p.xml")
    system = ff.createSystem(
        topology,
        constraints=app.HBonds,
        nonbondedMethod=app.NoCutoff,
        nonbondedCutoff=cutoff_distance,
    )
    positions = pdb.getPositions(asNumpy=True)
    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    return simulation


def generate_simulation(**kwargs):
    """
    Generates a simulation context, we pass this function to `pysages.run`.
    """
    print(f"string steps: {kwargs.get('stringstep')} replica: {kwargs.get('replica_num')}")
    simulation = internal_openmm_prep()

    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            10 * kwargs.get("log_delay", 100),
            step=True,
            remainingTime=True,
            totalSteps=(kwargs["string_steps"]) * kwargs["time_steps"],
        )
    )
    simulation.loadState(
        os.path.abspath(
            os.path.join(
                "string_conf", f"{kwargs.get('stringstep')}-{kwargs.get('replica_num')}.xml"
            )
        )
    )
    return simulation


def get_args(argv):
    available_args = [
        ("k-spring", "k", float, 50, "Spring constant for each replica"),
        ("replicas", "N", int, 16, "Number of replicas along the path"),
        ("time-steps", "t", int, 5e4, "Number of simulation steps for each replica"),
        ("log-period", "l", int, 100, "Frequency of logging the CVs into each histogram"),
        ("log-delay", "d", int, 100, "Number of timesteps to discard before logging"),
        ("string-steps", "s", int, 25, "Number of string iterations before finishing."),
        ("alpha", "a", float, 1e-1, "Update step size of the string update."),
    ]
    parser = argparse.ArgumentParser(
        description="Example script to run the spline (improved) string method"
    )
    for name, short, T, val, doc in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    parser.add_argument("--mpi", action="store_true", help="Use MPI executor")
    args = parser.parse_args(argv)
    return args


def get_executor(args):
    if args.mpi:
        futures = importlib.import_module("mpi4py.futures")
        return futures.MPIPoolExecutor()
    return SerialExecutor()


def prep_start(args):
    shutil.rmtree("string_conf", ignore_errors=True)
    os.mkdir("string_conf")
    simulation = internal_openmm_prep()
    simulation.minimizeEnergy()

    for i in range(args.replicas):
        simulation.saveState(os.path.join("string_conf", f"0-{i}.xml"))


def post_run_action(**kwargs):
    kwargs.get("context").saveState(
        os.path.join("string_conf", f"{kwargs.get('stringstep')+1}-{kwargs.get('replica_num')}.xml")
    )


def plot_energy(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("$s$")
    ax.set_ylabel("Free energy $[\\epsilon]$", color="teal")
    ax2 = ax.twinx()
    ax2.set_ylabel("convergence", color="maroon")
    s = np.linspace(0, 1, len(result["point_convergence"]))
    free_energy = np.asarray(result["free_energy"])
    offset = np.min(free_energy)

    ax.plot(s, free_energy - offset, "o-", color="teal")
    ax2.plot(s, result["point_convergence"], color="maroon")

    fig.savefig("energy.pdf", transparent=True, bbox_inches="tight", pad_inches=0)
    return fig


def plot_path(result):
    fig, ax = plt.subplots()
    ax.set_ylabel(r"$\phi$")
    ax.set_xlabel(r"$\psi$")
    ax.set_ylim((-np.pi, np.pi))
    ax.set_xlim((-np.pi, np.pi))

    path_history = np.asarray(result["path_history"])
    time = []
    segments = []
    for i in range(path_history.shape[0]):
        x = path_history[i, :, 1]
        y = path_history[i, :, 0]
        segments.append(np.column_stack([x, y]))
        time.append(i)
    lc = matplotlib.collections.LineCollection(segments)
    lc.set_array(np.asarray(time))
    ax.add_collection(lc)
    axcb = fig.colorbar(lc)
    axcb.set_label("String Iteration")

    fig.savefig("path_hist.pdf", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main(argv):
    args = get_args(argv)

    prep_start(args)
    cvs = (DihedralAngle((4, 6, 8, 14)), DihedralAngle((6, 8, 14, 16)))
    centers = []
    center_pos = np.linspace(+0.45 * np.pi, -0.45 * np.pi, args.replicas)
    for pos in center_pos:
        centers.append((pos, pos))
    method = SplineString(cvs, args.k_spring, centers, args.alpha, args.log_period, args.log_delay)
    context_args = vars(args)

    raw_result = pysages.run(
        method,
        generate_simulation,
        args.time_steps,
        args.string_steps,
        context_args=context_args,
        executor=get_executor(args),
        post_run_action=post_run_action,
    )
    result = pysages.analyze(raw_result)
    plot_path(result)
    plot_energy(result)

    pysages.save(result, "result.pkl")


if __name__ == "__main__":
    main(sys.argv[1:])
