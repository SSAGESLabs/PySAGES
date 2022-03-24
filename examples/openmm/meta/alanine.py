#!/usr/bin/env python3


# %%
from pysages.collective_variables import DihedralAngle
from pysages.methods import meta
from pysages.utils import try_import

import numpy
import pysages

import time

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


# %%
pi = numpy.pi


# %%
def generate_simulation(
    pdb_filename="../../../../inputs/alanine-dipeptide/adp.pdb",
    T=298.15 * unit.kelvin,
    dt=2.0 * unit.femtoseconds,
):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml")
    # kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    # kT = (kB * T).value_in_unit(unit.kilojoules_per_mole)
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
def main():
    cvs = (DihedralAngle((4, 6, 8, 14)), DihedralAngle((6, 8, 14, 16)))

    angle = numpy.array([True, True])

    height = 1.2
    sigma = numpy.array([[0.35, 0.35]])
    delta_T = -1  # 1500
    stride = 500
    timesteps = int(5000)  # int(1e8)
    nstrides = timesteps // stride + 1
    hillsFile = "hills.dat"
    colvarFile = "colvarFile.dat"
    colvar_outstride = 100

    grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(50, 50), periodic=True)

    method = meta(
        cvs,
        height,
        sigma,
        stride,
        nstrides,
        delta_T,
        colvar_outstride=colvar_outstride,
        angle=angle,
    )
    # grid = grid
    # , hillsFile=hillsFile, colvarFile=colvarFile)

    tic = time.perf_counter()
    method.run(generate_simulation, timesteps)
    toc = time.perf_counter()
    print(f"Completed the simulation in {toc - tic:0.4f} seconds.")


# %%
if __name__ == "__main__":
    main()
