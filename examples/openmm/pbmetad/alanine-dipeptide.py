#!/usr/bin/env python3

"""
Parallel bias metadynamics simulation of Alanine Dipeptide in vacuum with OpenMM and PySAGES.

Example command to run the simulation `python3 alanine-dipeptide.py --time-steps 1000`
For other supported commandline parameters, check `python3 alanine-dipeptide.py --help`
"""


# %%
import argparse
import os
import sys
import time

import numpy
import pysages

from pysages.colvars import DihedralAngle
from pysages.methods import ParallelBiasMetadynamics, MetaDLogger
from pysages.utils import try_import
from pysages.approxfun import compute_mesh

import matplotlib.pyplot as plt

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


# %%
pi = numpy.pi
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
kB = kB.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)

T_val = 298.15
T = T_val * unit.kelvin
dt = 2.0 * unit.femtoseconds
adp_pdb = os.path.join(os.pardir, os.pardir, "inputs", "alanine-dipeptide", "adp-vacuum.pdb")


# %%
def generate_simulation(pdb_filename=adp_pdb, T=T, dt=dt):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml")
    cutoff_distance = 1.0 * unit.nanometer
    topology = pdb.topology

    system = ff.createSystem(
        topology, constraints=app.HBonds, nonbondedMethod=app.PME, nonbondedCutoff=cutoff_distance
    )

    # Set dispersion correction use.
    forces = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        forces[force.__class__.__name__] = force

    forces["NonbondedForce"].setUseDispersionCorrection(True)
    forces["NonbondedForce"].setEwaldErrorTolerance(1.0e-5)

    positions = pdb.getPositions(asNumpy=True)

    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)

    integrator.setRandomNumberSeed(42)

    # platform = openmm.Platform.getPlatformByName(platform)
    # simulation = app.Simulation(topology, system, integrator, platform)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    simulation.reporters.append(app.PDBReporter("output.pdb", 1000))
    simulation.reporters.append(
        app.StateDataReporter("log.dat", 1000, step=True, potentialEnergy=True, temperature=True)
    )

    return simulation


# %%
def get_args(argv):
    available_args = [
        ("well-tempered", "w", bool, 0, "Whether to use well-tempered metadynamics"),
        ("use-grids", "g", bool, 0, "Whether to use grid acceleration"),
        ("log", "l", bool, 0, "Whether to use a callback to log data into a file"),
        ("time-steps", "t", int, 5e5, "Number of simulation steps"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run metadynamics")
    for (name, short, T, val, doc) in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    return parser.parse_args(argv)


# %%
def main(argv=[]):
    args = get_args(argv)

    cvs = [DihedralAngle([4, 6, 8, 14]), DihedralAngle([6, 8, 14, 16])]

    height = [1.2, 1.2]  # kJ/mol
    sigma = [0.35, 0.35]  # radians
    deltaT = 5000
    stride = 500  # frequency for depositing gaussians
    timesteps = args.time_steps
    ngauss = timesteps // stride + 1  # total number of gaussians

    # Grid for storing bias potential and its gradient
    grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(50, 50), periodic=True)
    grid = grid if args.use_grids else None

    # Method
    method = ParallelBiasMetadynamics(
        cvs, height, sigma, stride, ngauss, T_val, deltaT, kB, grid=None
    )

    # Logging
    hills_file = "hills.dat"
    callback = MetaDLogger(hills_file, stride) if args.log else None

    tic = time.perf_counter()
    run_result = pysages.run(method, generate_simulation, timesteps, callback)
    toc = time.perf_counter()
    print(f"Completed the simulation in {toc - tic:0.4f} seconds.")

    ####   # Analysis: Calculate free energy using the deposited bias potential
    ####
    ####   # generate CV values on a grid to evaluate bias potential
    ####   plot_grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(64, 64), periodic=True)
    ####   xi = (compute_mesh(plot_grid) + 1) / 2 * plot_grid.size + plot_grid.lower
    ####
    ####   # determine bias factor depending on method (for standard = 1 and for well-tempered = (T+deltaT)/deltaT)
    ####   alpha = (
    ####       1
    ####       if method.deltaT is None
    ####       else (T.value_in_unit(unit.kelvin) + method.deltaT) / method.deltaT
    ####   )
    ####   kT = kB * T.value_in_unit(unit.kelvin)
    ####
    ####   # extract metapotential function from result
    ####   result = pysages.analyze(run_result)
    ####   metapotential = result["metapotential"]
    ####
    ####   # report in kT and set min free energy to zero
    ####   A = metapotential(xi) * -alpha / kT
    ####   A = A - A.min()
    ####   A = A.reshape(plot_grid.shape)
    ####
    ####   # plot and save free energy to a PNG file
    ####   fig, ax = plt.subplots(dpi=120)
    ####
    ####   im = ax.imshow(A, interpolation="bicubic", origin="lower", extent=[-pi, pi, -pi, pi])
    ####   ax.contour(A, levels=12, linewidths=0.75, colors="k", extent=[-pi, pi, -pi, pi])
    ####   ax.set_xlabel(r"$\phi$")
    ####   ax.set_ylabel(r"$\psi$")
    ####
    ####   cbar = plt.colorbar(im)
    ####   cbar.ax.set_ylabel(r"$A~[k_{B}T]$", rotation=270, labelpad=20)
    ####
    ####   fig.savefig("adp-fe.png", dpi=fig.dpi)

    return run_result


# %%
if __name__ == "__main__":
    main(sys.argv[1:])
