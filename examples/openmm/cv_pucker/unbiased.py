#!/usr/bin/env python

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

import pysages
from pysages.colvars.angles import RingPuckeringCoordinates
from pysages.methods import HistogramLogger, Unbiased

step_size = 1 * unit.femtosecond
nsteps = 100
print("nsteps = {}".format(nsteps))

pdb = app.PDBFile("guanosine.fixed.pdb")
sugar_indices = {}
for i, atom in enumerate(pdb.topology.atoms()):
    if atom.name in ["O4'", "C4'", "C3'", "C2'", "C1'"]:
        sugar_indices[atom.name] = atom.index

print(f"dictionary of sugar indices: {sugar_indices}")


def generate_simulation():

    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometer,
        constraints=app.HBonds,
    )
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, 298 * unit.kelvin, 1000))

    integrator = mm.LangevinIntegrator(
        298 * unit.kelvin, 5 / unit.picosecond, step_size.in_units_of(unit.picosecond)
    )
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    print("minimizing energy...")
    simulation.minimizeEnergy()

    print("Using {} platform".format(simulation.context.getPlatform().getName()))
    simulation.reporters.append(app.DCDReporter("output.dcd", 1))

    return simulation


cvs = [RingPuckeringCoordinates([sugar_indices[nm] for nm in ["O4'", "C1'", "C2'", "C3'", "C4'"]])]
# notice that the order of the indices do matter for the calculation of phase angle
method = Unbiased(cvs)
callback = HistogramLogger(1)

raw_result = pysages.run(method, generate_simulation, nsteps, callback)
np.savetxt("phase_angle.txt", raw_result.callbacks[0].data)
