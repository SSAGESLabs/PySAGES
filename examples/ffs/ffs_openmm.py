#!/usr/bin/env python3


# %%
from pysages.collective_variables import DihedralAngle
from pysages.methods import FFS

import simtk.openmm.app as app
import simtk.openmm as openmm
import simtk.unit as unit
import jax.numpy as np
import numpy
import pysages


# %%
pi = numpy.pi


# %%
def generate_simulation(
    pdb_filename = "../inputs/alanine-dipeptide/adp-explicit.pdb",
    T = 298.15 * unit.kelvin,
    dt = 2.0 * unit.femtoseconds
):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml", "tip3p.xml")
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = (kB * T).value_in_unit(unit.kilojoules_per_mole)
    cutoff_distance = 1.0 * unit.nanometer
    topology = pdb.topology

    system = ff.createSystem(
        topology, constraints = app.HBonds, nonbondedMethod = app.PME,
        nonbondedCutoff = cutoff_distance
    )

    # Set dispersion correction use.
    forces = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        forces[force.__class__.__name__] = force

    forces["NonbondedForce"].setUseDispersionCorrection(True)
    forces["NonbondedForce"].setEwaldErrorTolerance(1.0e-5)

    positions = pdb.getPositions(asNumpy = True)

    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)

    #platform = openmm.Platform.getPlatformByName(platform)
    #simulation = app.Simulation(topology, system, integrator, platform)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    return simulation


# %%
def main():
    cvs = (DihedralAngle((6, 8, 14, 16)),)
    grid = pysages.Grid(
    lower = ((-5.0,)),
    upper = ((5.0,)),
    shape = ((25,)),
    periodic = False
    )
    dt=2.0
    method = FFS(cvs,grid)
    win_0=(100./180.)*np.pi
    win_f=(150./180.)*np.pi
    #Here there will be different from other methods
    method.run(generate_simulation,1e9,dt,win_0,win_f,4,20000,20)


# %%
if __name__ == "__main__":
    main()
