#!/usr/bin/env python3


# %%
import numpy as np

import ase.units as units
from ase import Atoms
from ase.constraints import FixBondLengths
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.md import Langevin
from ase.io.trajectory import Trajectory

import pysages
from pysages.grids import Grid
from pysages.colvars import Distance
from pysages.methods import ABF


# %%
def generate_simulation(tag="tip3p"):
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
    logfile = tag + ".log"
    atoms.calc = TIP3P(rc=4.5)
    md = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.01, logfile=logfile)

    traj = Trajectory(tag + ".traj", "w", atoms)
    md.attach(traj.write, interval=1)

    return md


# %%
def main():
    cvs = [Distance([0, 3])]
    grid = Grid(lower=0.1, upper=9.0, shape=64)
    method = ABF(cvs, grid)
    pysages.run(method, generate_simulation, 100)


# %%
if __name__ == "__main__":
    main()
