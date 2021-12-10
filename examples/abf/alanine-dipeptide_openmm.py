#!/usr/bin/env python3


# %%
from pysages.collective_variables import DihedralAngle
from pysages.methods import ABF
from pysages.utils import try_import

import numpy
import pysages

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


# %%
pi = numpy.pi


# %%
def generate_simulation(
    pdb_filename = "alanine-dipeptide-explicit.pdb",
    T = 298.15 * unit.kelvin,
    dt = 2.0 * unit.femtoseconds
):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml", "tip3p.xml")
    # kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    # kT = (kB * T).value_in_unit(unit.kilojoules_per_mole)
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

    # platform = openmm.Platform.getPlatformByName(platform)
    # simulation = app.Simulation(topology, system, integrator, platform)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    return simulation


# %%
def main():
    cvs = (
        DihedralAngle((4, 6, 8, 14)),
        DihedralAngle((6, 8, 14, 16))
    )

    grid = pysages.Grid(
        lower = (-pi, -pi),
        upper = (pi, pi),
        shape = (32, 32),
        periodic = True
    )

    method = ABF(cvs, grid)

    method.run(generate_simulation, 50)


# %%
if __name__ == "__main__":
    main()
