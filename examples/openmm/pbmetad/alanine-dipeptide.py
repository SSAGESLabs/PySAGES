#!/usr/bin/env python3

"""
Parallel Bias Well-tempered Metadynamics (PBMetaD) simulation of 
Alanine Dipeptide along backbone dihedrals in vacuum with OpenMM and PySAGES.

Example command to run the simulation `python3 alanine-dipeptide.py --time_steps 1000`
For other supported commandline parameters, check `python3 alanine-dipeptide.py --help`
"""


# %%
import argparse
import os
import sys
import time

import numpy
import pysages
from jax import numpy as np

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

T = 298.15 * unit.kelvin
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
        ("use_grids", "g", bool, 0, "Whether to use grid acceleration"),
        ("log", "l", bool, 0, "Whether to use a callback to log data into a file"),
        ("time_steps", "t", int, 5e5, "Number of simulation steps"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run metadynamics")
    for (name, short, T, val, doc) in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    return parser.parse_args(argv)


# %%
def main(argv=[]):
    args = get_args(argv)
    print(args)

    cvs = [DihedralAngle([4, 6, 8, 14]), DihedralAngle([6, 8, 14, 16])]

    height = [1.2, 1.2]  # kJ/mol
    sigma = [0.35, 0.35]  # radians
    deltaT = 5000
    stride = 500  # frequency for depositing gaussians
    timesteps = args.time_steps
    ngauss = timesteps // stride + 1  # total number of gaussians

    # Grid for storing bias potential and its gradient
    grid = pysages.Grid(
        lower=(-pi, -pi), upper=(pi, pi), shape=(50, 50), periodic=True, parallelbias=True
    )

    grid = grid if args.use_grids else None

    # Method
    method = ParallelBiasMetadynamics(
        cvs, height, sigma, stride, ngauss, T.value_in_unit(unit.kelvin), deltaT, kB, grid=grid
    )

    # Logging
    hills_file = "hills.dat"
    callback = MetaDLogger(hills_file, stride) if args.log else None

    tic = time.perf_counter()
    run_result = pysages.run(method, generate_simulation, timesteps, callback)
    toc = time.perf_counter()
    print(f"Completed the simulation in {toc - tic:0.4f} seconds.")

    # Analysis: Calculate free energy using the deposited bias potential

    # generate CV values on a grid to evaluate bias potential
    plot_1Dgrid = pysages.Grid(lower=(-pi), upper=(pi), shape=(64), periodic=True)

    xi_1D = (compute_mesh(plot_1Dgrid) + 1) / 2 * plot_1Dgrid.size + plot_1Dgrid.lower
    xi_1D_2CVs = np.hstack((xi_1D, xi_1D))

    # determine bias factor = (T+deltaT)/deltaT)
    alpha = (T.value_in_unit(unit.kelvin) + method.deltaT) / method.deltaT
    kT = kB * T.value_in_unit(unit.kelvin)
    beta = 1 / kT

    # extract metapotential function from result
    result = pysages.analyze(run_result)
    centers = result["centers"]
    heights = result["heights"]

    pbmetad_potential_cv = result["pbmetad_potential_cv"]
    pbmetad_net_potential = result["pbmetad_net_potential"]

    # calculate free energy and report in kT at the end of simulation.
    A_cv = pbmetad_potential_cv(xi_1D_2CVs) * -alpha / kT
    # set min free energy to zero
    A_cv = A_cv - A_cv.min(axis=0)
    A_cv1 = A_cv[:, 0].reshape(plot_1Dgrid.shape)
    A_cv2 = A_cv[:, 1].reshape(plot_1Dgrid.shape)

    # plot and save free energy along each CV to a PNG file
    fig = plt.figure(figsize=(8, 8), dpi=120)
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    range_to_time = stride * dt.value_in_unit(unit.femtoseconds) * 1e-6

    # plot centers along phi and psi to monitor sampling
    for i in range(2):
        ax = fig.add_subplot(3, 2, i + 1)
        color = "blue" if i == 0 else "red"
        ylabel = r"$\phi$" if i == 0 else r"$\psi$"
        ax.scatter(
            np.arange(np.shape(centers)[0]) * range_to_time, centers[:, i], s=20, color=color
        )
        ax.set_xlabel(r"Time [ns]")
        ax.set_ylabel(ylabel)
        ax.set_yticks(np.arange(-pi, pi + pi / 2, step=(pi / 2)), ["-π", "-π/2", "0", "π/2", "π"])

    # plot height along phi and psi to monitor sampling
    for i in range(2):
        ax = fig.add_subplot(3, 2, i + 3)
        color = "blue" if i == 0 else "red"
        ylabel = r"$W(\phi) ~[k_{B}T]$" if i == 0 else r"$W(\psi) ~[k_{B}T]$"
        ax.scatter(
            np.arange(np.shape(heights)[0]) * range_to_time, heights[:, i], s=20, color=color
        )
        ax.set_xlabel(r"Time [ns]")
        ax.set_ylabel(ylabel)

    # plot free energy along phi and psi at the end of simulation
    for i in range(2):
        ax = fig.add_subplot(3, 2, i + 5)
        color = "blue" if i == 0 else "red"
        xlabel = r"$\phi$" if i == 0 else r"$\phi$"
        y = A_cv1 if i == 0 else A_cv2
        ax.plot(xi_1D, y, lw=3, color=color)
        ax.set_xticks(np.arange(-pi, pi + pi / 2, step=(pi / 2)), ["-π", "-π/2", "0", "π/2", "π"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$A~[k_{B}T]$")

    fig.savefig("adp-fe.png", dpi=fig.dpi)

    return result


# %%
if __name__ == "__main__":
    main(sys.argv[1:])
