#!/usr/bin/env python3

# %%
import argparse
import sys

import numpy as np
from ase import Atoms, units
from ase.calculators.tip3p import TIP3P, angleHOH, rOH
from ase.constraints import FixBondLengths
from ase.io.trajectory import Trajectory
from ase.md import Langevin

import pysages
from pysages.colvars import Angle
from pysages.grids import Grid
from pysages.methods import ABF


# %%
def generate_simulation(tag="tip3p", write_output=True):
    x = angleHOH * np.pi / 180 / 2
    pos = [
        [0, 0, 0],  # rOH is the distance between oxygen and hydrogen atoms in water
        [0, rOH * np.cos(x), rOH * np.sin(x)],
        [0, rOH * np.cos(x), -rOH * np.sin(x)],
    ]
    atoms = Atoms("OH2", positions=pos)

    vol = ((18.01528 / 6.022140857e23) / (0.9982 / 1e24)) ** (1 / 3.0)
    atoms.set_cell((vol, vol, vol))
    atoms.center()

    atoms = atoms.repeat((3, 3, 3))
    atoms.set_pbc(True)

    atoms.constraints = FixBondLengths(
        [(3 * i + j, 3 * i + (j + 1) % 3) for i in range(3**3) for j in [0, 1, 2]]
    )

    T = 300 * units.kB
    atoms.calc = TIP3P(rc=4.5)
    logfile = tag + ".log" if write_output else None
    md = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.01, logfile=logfile)

    if write_output:
        traj = Trajectory(tag + ".traj", "w", atoms)
        md.attach(traj.write, interval=1)

    return md


# %%
def process_args(argv):
    print(repr(argv))
    available_args = [
        ("timesteps", "t", int, 100, "Number of simulation steps"),
        ("write-output", "o", bool, 1, "Write log and trajectory of the ASE run"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run pysages with ASE")

    for name, short, T, val, doc in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)

    return parser.parse_args(argv)


# %%
def run_simulation(timesteps, write_output):
    cvs = [Angle([1, 0, 2])]
    grid = Grid(lower=0.1, upper=9.0, shape=64, periodic=True)
    method = ABF(cvs, grid)
    context_args = dict(write_output=write_output)
    return pysages.run(method, generate_simulation, timesteps, context_args=context_args)


# %%
def main(argv=None):
    args = process_args([] if argv is None else argv)
    run_simulation(args.timesteps, args.write_output)


# %%
if __name__ == "__main__":
    main(sys.argv[1:])
