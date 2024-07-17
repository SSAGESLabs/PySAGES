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

AUGC = ["A", "U", "G", "C"]


def create_exclusions_from_bonds(particles, bonds, bond_cutoff=3):
    """
    create exclusion from bond.
    """
    n_particles = max(particles) + 1
    exclusions = [set() for _ in range(n_particles)]
    bonded12 = [set() for _ in range(n_particles)]
    for bond in bonds:
        p1, p2 = bond
        exclusions[p1].add(p2)
        exclusions[p2].add(p1)
        bonded12[p1].add(p2)
        bonded12[p2].add(p1)

    for level in range(bond_cutoff - 1):
        current_exclusions = [exclusion.copy() for exclusion in exclusions]
        for i in range(n_particles):
            for j in current_exclusions[i]:
                exclusions[j].update(bonded12[i])

    final_exclusions = []
    for i in range(len(exclusions)):
        for j in exclusions[i]:
            if j < i:
                final_exclusions.append((j, i))

    return final_exclusions


step_size = 2 * unit.femtosecond
nsteps = 100

contact_cutoff = 0.5  # nanometer

pdb = app.PDBFile("../../inputs/GAGA.box_0mM.pdb")
positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

rna_indices = []
for atom in pdb.topology.atoms():
    if atom.residue.name in AUGC and atom.element.name != "hydrogen":
        rna_indices.append(atom.index)

rna_bonds = []
for bond in pdb.topology.bonds():
    if (
        bond.atom1.residue.name in AUGC
        and bond.atom1.element.name != "hydrogen"
        and bond.atom2.residue.name in AUGC
        and bond.atom2.element.name != "hydrogen"
    ):
        rna_bonds.append([bond.atom1.index, bond.atom2.index])

exclusions = create_exclusions_from_bonds(rna_indices, rna_bonds)

rna_pos = positions.astype("float")[np.asarray(rna_indices)]
contact_matrix = sd.squareform(sd.pdist(rna_pos)) < contact_cutoff
contacts = np.transpose(np.nonzero(contact_matrix))
rna_id_contacts = np.array(
    [
        [rna_indices[i], rna_indices[j]]
        for i, j in contacts
        if i != j
        and (rna_indices[i], rna_indices[j]) not in exclusions
        and (rna_indices[j], rna_indices[i]) not in exclusions
    ]
)
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
