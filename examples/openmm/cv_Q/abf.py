#!/usr/bin/env python

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit
from scipy.spatial import distance as sd

import pysages
from pysages import Grid
from pysages.colvars.contacts import NativeContactFraction
from pysages.methods import ABF, HistogramLogger

step_size = 2 * unit.femtosecond
nsteps = 100

contact_cutoff = 0.5  # nanometer

pdb = app.PDBFile("../../inputs/GAGA.box_0mM.pdb")
positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
rna_indices = []
for i, residue in enumerate(pdb.topology.residues()):
    if residue.name in ["A", "U", "G", "C"]:
        for atom in residue.atoms():
            if atom.element.name != "hydrogen":
                rna_indices.append(atom.index)

rna_pos = positions.astype("float")[np.asarray(rna_indices)]
contact_matrix = sd.squareform(sd.pdist(rna_pos)) < contact_cutoff
contacts = np.transpose(np.nonzero(contact_matrix))
rna_id_contacts = np.array([[rna_indices[i], rna_indices[j]] for i, j in contacts if i != j])
# notice that we need to get rid of self-self contact!
indices = np.unique(rna_id_contacts)
references = positions.astype("float")[np.asarray(indices)]


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
    simulation.reporters.append(app.PDBReporter("output.pdb", 1, enforcePeriodicBox=False))
    simulation.reporters.append(
        app.StateDataReporter(
            "log",
            1,
            step=True,
            time=True,
            speed=True,
            remainingTime=True,
            elapsedTime=True,
            totalSteps=nsteps,
        )
    )

    return simulation


def main():
    cvs = [
        NativeContactFraction(indices, rna_id_contacts, references),
    ]

    method = ABF(cvs, Grid(lower=(0,), upper=(1,), shape=(32,)))
    callback = HistogramLogger(1)

    raw_result = pysages.run(method, generate_simulation, nsteps, callback)
    pysages.save(raw_result, "state.pkl")

    np.savetxt("Q.txt", raw_result.callbacks[0].data[:, :1])
    np.savetxt("references.txt", references)
    np.save("contact_pairs.npy", rna_id_contacts)
    np.save("contact_pairs_remapped.npy", cvs[0].contact_pairs)


if __name__ == "__main__":
    main()
