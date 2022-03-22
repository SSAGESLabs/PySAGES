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

adp_pdb = "../inputs/alanine-dipeptide/adp-explicit.pdb"
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
    parser = argparse.ArgumentParser(description="Run forward flux sampling.")
    parser.add_argument(
        "--timesteps",
        metavar="t",
        type=int,
        default=int(1e9),
        help="Max number of timesteps",
        required=False,
    )
    parser.add_argument(
        "--window_number",
        metavar="Nw",
        type=int,
        default=4,
        help="Number of windows.",
        required=False,
    )
    parser.add_argument(
        "--sampling_steps",
        metavar="S",
        type=int,
        default=20000,
        help="Period for sampling configurations in the basin.",
        required=False,
    )
    parser.add_argument(
        "--replicas",
        metavar="R",
        type=int,
        default=20,
        help="Number of stored configurations for each window.",
        required=False,
    )
    args = parser.parse_args()

    cvs = [DihedralAngle((6, 8, 14, 16))]
    method = FFS(cvs)

    dt = 2.0
    win_0 = (100 / 180) * pi
    win_f = (150 / 180) * pi

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
