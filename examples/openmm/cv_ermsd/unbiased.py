#!/usr/bin/env python

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

import pysages
from pysages.colvars.coordinates import Component
from pysages.colvars.orientation import ERMSD
from pysages.methods import HistogramLogger, Unbiased

step_size = 2 * unit.femtosecond
nsteps = 100

pdb = app.PDBFile("../../inputs/GAGA.box_0mM.pdb")
C246_indices = []
for i, res in enumerate(pdb.topology.residues()):
    C246_residue = dict.fromkeys(["C2", "C4", "C6"])
    for atom in res.atoms():
        if atom.name in ["C2", "C4", "C6"]:
            C246_residue[atom.name] = atom.index
    C246_indices.append(C246_residue)

# notice that the order of the indices for eRMSD is tricky!
C246_indices_ordered = []
for i, res in enumerate(pdb.topology.residues()):
    if res.name in ["G", "A"]:
        C246 = C246_indices[i]
        C246_indices_ordered.extend((C246["C2"], C246["C6"], C246["C4"]))
    elif res.name in ["U", "C"]:
        C246 = C246_indices[i]
        C246_indices_ordered.extend((C246["C2"], C246["C4"], C246["C6"]))

reference = pdb.getPositions(asNumpy=True).astype("float")[np.asarray(C246_indices_ordered)]


def generate_simulation():
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometer,
        constraints=app.HBonds,
    )

    integrator = mm.LangevinIntegrator(
        298 * unit.kelvin, 5 / unit.picosecond, step_size.in_units_of(unit.picosecond)
    )
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    print("minimizing energy...")
    simulation.minimizeEnergy()

    print("Using {} platform".format(simulation.context.getPlatform().getName()))
    simulation.reporters.append(app.DCDReporter("output.dcd", 1, enforcePeriodicBox=False))

    return simulation


def main():
    cvs = [
        ERMSD(C246_indices_ordered, reference, cutoff=3.2),
    ]

    for idx in C246_indices_ordered:
        for i in range(3):
            cvs += [Component([idx], axis=i)]
            # record the position of atom C2/4/6 in the base for testing

    method = Unbiased(cvs)
    callback = HistogramLogger(1)

    raw_result = pysages.run(method, generate_simulation, nsteps, callback)
    np.savetxt("ermsd.txt", raw_result.callbacks[0].data[:, :1])
    np.savetxt("base_coords.txt", raw_result.callbacks[0].data[:, 1:])
    np.savetxt("reference.txt", reference)


if __name__ == "__main__":
    main()
