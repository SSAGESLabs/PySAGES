#!/usr/bin/env python3


# %%
from pysages.collective_variables import DihedralAngle
from pysages.methods import FFS
from pysages.utils import try_import

import argparse
import numpy
import pysages

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


# %%
pi = numpy.pi
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

adp_pdb = "../inputs/alanine-dipeptide/adp-vacuum.pdb"
T = 298.15 * unit.kelvin
dt = 2.0 * unit.femtoseconds


# %%
def generate_simulation(pdb_filename=adp_pdb, T=T, dt=dt):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml", "tip3p.xml")
    kT = (kB * T).value_in_unit(unit.kilojoules_per_mole)
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

    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    return simulation


# %%
def main():
    available_args = [
        ("timesteps", "t", int, 1e9, "Max number of timesteps."),
        ("cv-start", "o", float, 100, "Intial value of the dihedral."),
        ("cv-distance", "d", float, 50, "Distance from the intial to the final dihedral."),
        ("window-number", "Nw", int, 4, "Number of windows."),
        ("sampling-steps", "S", int, 20000, "Period for sampling configurations in the basin."),
        ("replicas", "R", int, 20, "Number of stored configurations for each window."),
    ]
    parser = argparse.ArgumentParser(description="Run forward flux sampling.")
    for (name, short, T, val, doc) in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    args = parser.parse_args()

    cvs = [DihedralAngle((6, 8, 14, 16))]
    method = FFS(cvs)

    dt = 2.0
    win_0 = (args.cv_start / 180) * pi
    win_f = ((args.cv_start + args.cv_distance) / 180) * pi

    method.run(
        generate_simulation,
        args.timesteps,
        dt,
        win_0,
        win_f,
        args.window_number,
        args.sampling_steps,
        args.replicas,
    )


# %%
if __name__ == "__main__":
    main()
