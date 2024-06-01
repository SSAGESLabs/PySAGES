#!/usr/bin/env python

import jax
import numpy as np
import openmm.app as app
from jax import grad

from pysages.colvars.orientation import ERMSD, RMSD

pdb = app.PDBFile("../../../inputs/GAGA.box_0mM.pdb")

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

ermsd = ERMSD(C246_indices_ordered, reference)
ermsd_grad = grad(ermsd.function)
print(ermsd_grad(reference + np.random.random(reference.shape)))

rmsd = RMSD(C246_indices_ordered, reference)
rmsd_grad = grad(rmsd.function)
print(rmsd_grad(reference + np.random.random(reference.shape)))


def G(r, cutoff=2.4):
    gamma = jax.numpy.pi / cutoff
    end = jax.numpy.sin(gamma * r) * jax.numpy.heaviside(cutoff - r, np.zeros(r.shape))
    return end[0]


print(grad(G)(np.ones((10,))))
