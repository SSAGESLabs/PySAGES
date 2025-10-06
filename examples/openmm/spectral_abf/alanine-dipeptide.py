#!/usr/bin/env python3

"""
SpectralABF simulation of Alanine Dipeptide in vacuum with OpenMM and PySAGES.

Example command to run the simulation `python3 alanine-dipeptide.py --time-steps 1000`
For other supported commandline parameters, check `python3 alanine-dipeptide.py --help`
"""


# %%
import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy

import pysages
from pysages.approxfun import compute_mesh
from pysages.colvars import DihedralAngle
from pysages.methods import SpectralABF
from pysages.utils import try_import

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


# %%
pi = numpy.pi
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
kB = kB.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)

T = 298.15 * unit.kelvin
dt = 2.0 * unit.femtoseconds
adp_pdb = "adp-vacuum.pdb"


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

    return simulation


# %%
def get_args(argv):
    available_args = [
        ("time-steps", "t", int, 5e5, "Number of simulation steps"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run Spectral ABF")
    for name, short, T, val, doc in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    return parser.parse_args(argv)


# %%
def main(argv=[]):
    args = get_args(argv)

    cvs = [DihedralAngle([4, 6, 8, 14]), DihedralAngle([6, 8, 14, 16])]
    grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(50, 50), periodic=True)

    # Method
    method = SpectralABF(cvs, grid)

    tic = time.perf_counter()
    run_result = pysages.run(method, generate_simulation, args.time_steps)
    toc = time.perf_counter()
    print(f"Completed the simulation in {toc - tic:0.4f} seconds.")

    # Analysis: Calculate free energy
    result = pysages.analyze(run_result)
    fes_fn = result["fes_fn"]

    # generate CV values on a grid to evaluate bias potential
    plot_grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(64, 64), periodic=True)
    xi = compute_mesh(plot_grid)

    # Set min free energy to zero
    A = fes_fn(xi)
    A = A.reshape(plot_grid.shape)

    # plot and save free energy to a PNG file
    fig, ax = plt.subplots(dpi=120)

    im = ax.imshow(A, interpolation="bicubic", origin="lower", extent=[-pi, pi, -pi, pi])
    ax.contour(A, levels=12, linewidths=0.75, colors="k", extent=[-pi, pi, -pi, pi])
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")

    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel(r"$A~[kJ/mol]$", rotation=270, labelpad=20)

    fig.savefig("adp-fe.png", dpi=fig.dpi)

    return result


# %%
if __name__ == "__main__":
    main(sys.argv[1:])
