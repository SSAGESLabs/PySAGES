#!/usr/bin/env python3

import sys
import argparse
import importlib
import numpy as np

from pysages.colvars import DihedralAngle
from pysages.methods import UmbrellaIntegration, SerialExecutor
from pysages.utils import try_import

import pysages

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


def generate_simulation(**kwargs):
    """
    Generates a simulation context, we pass this function to `pysages.run`.
    """
    pdb_filename = "adp-explicit.pdb"
    T = 298.15 * unit.kelvin
    dt = 2.0 * unit.femtoseconds
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml", "tip3p.xml")
    cutoff_distance = 1.0 * unit.nanometer
    topology = pdb.topology
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
    simulation.minimizeEnergy()

    return simulation


def get_args(argv):
    available_args = [
        ("k-spring", "k", float, 50, "Spring constant for each replica"),
        ("replicas", "N", int, 25, "Number of replicas along the path"),
        ("time-steps", "t", int, 1e4, "Number of simulation steps for each replica"),
        ("log-period", "l", int, 100, "Frequency of logging the CVs into each histogram"),
        ("log-delay", "d", int, 0, "Number of timesteps to discard before logging"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run umbrella integration")
    for (name, short, T, val, doc) in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    parser.add_argument("--mpi", action="store_true", help="Use MPI executor")
    args = parser.parse_args(argv)
    return args


def get_executor(args):
    if args.mpi:
        futures = importlib.import_module("mpi4py.futures")
        return futures.MPIPoolExecutor()
    return SerialExecutor()


def main(argv):
    args = get_args(argv)

    cvs = (DihedralAngle((4, 6, 8, 14)), DihedralAngle((6, 8, 14, 16)))
    centers = []
    center_pos = np.linspace(+0.45 * np.pi, -0.45 * np.pi, args.replicas)
    for pos in center_pos:
        centers.append((pos, pos))
    method = UmbrellaIntegration(cvs, args.k_spring, centers, args.log_period, args.log_delay)
    raw_result = pysages.run(
        method,
        generate_simulation,
        args.time_steps,
        executor=get_executor(args),
    )
    result = pysages.analyze(raw_result)
    print(result)

    return result


if __name__ == "__main__":
    main(sys.argv[1:])
