#!/usr/bin/env python

from itertools import product

import networkx as nx
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


def init_graph(top):
    graph = nx.Graph()
    rna_indices = []
    full_id2rna_heavy_id = {}
    index = 0
    for atom in top.atoms():
        if atom.residue.name in AUGC and atom.element.name != "hydrogen":
            graph.add_node(index, name=atom.name, resid=atom.residue.index)
            full_id2rna_heavy_id[atom.index] = index
            rna_indices.append(atom.index)
            index += 1

    for bond in top.bonds():
        if (
            bond.atom1.residue.name in AUGC
            and bond.atom1.element.name != "hydrogen"
            and bond.atom2.residue.name in AUGC
            and bond.atom2.element.name != "hydrogen"
        ):
            graph.add_edge(
                full_id2rna_heavy_id[bond.atom1.index], full_id2rna_heavy_id[bond.atom2.index]
            )

    return graph, rna_indices


def gen_dihedral_indices(graph):
    dihedral_indices = []
    for n1, n2 in nx.edge_dfs(graph):
        neibor1 = list(graph.adj[n1])
        neibor1.remove(n2)
        # we want to exclude neightbor n2
        # because we are trying to construct a dihedral pre-n1-n2-post,
        # and we don't want pre is the same as n2
        neibor2 = list(graph.adj[n2])
        neibor2.remove(n1)
        for pre, post in product(neibor1, neibor2):
            if pre != post:  # exlucde ring dihedrals
                dihedral_indices.append([pre, n1, n2, post])
    return dihedral_indices


def calc_include_ij(graph, dihedral_indices):
    """
    calculate all the atom pairs that are connected by 1 or 2 or 3 bonds.
    Those pairs can't be included in the native contacts
    return a matrix for all the atoms in the graph:
        True means we can include the pair
        False means we should exclude the pair
    """
    n_atoms = len(graph)
    exclude_ij = np.full((n_atoms, n_atoms), False)
    for dihedral in dihedral_indices:
        indices = np.array(np.meshgrid(dihedral, dihedral)).T.reshape(-1, 2)
        i, j = indices[:, 0], indices[:, 1]
        exclude_ij[i, j] = True

    return ~exclude_ij


step_size = 2 * unit.femtosecond
nsteps = 100

contact_cutoff = 0.5  # nanometer

pdb = app.PDBFile("../../inputs/GAGA.box_0mM.pdb")
positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

rna_graph, rna_indices = init_graph(pdb.topology)
dihedral_indices = gen_dihedral_indices(rna_graph)
include_ij = calc_include_ij(rna_graph, dihedral_indices)

rna_pos = positions.astype("float")[np.asarray(rna_indices)]
contact_matrix = sd.squareform(sd.pdist(rna_pos)) < contact_cutoff
contacts = np.transpose(np.nonzero(contact_matrix))
rna_id_contacts = np.array(
    [[rna_indices[i], rna_indices[j]] for i, j in contacts if i != j and include_ij[i, j]]
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
