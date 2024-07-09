#!/usr/bin/env python

import numpy as np
import openmm.app as app
from jax import grad
from openmm import unit
from scipy.spatial import distance as sd

from pysages.colvars.contacts import NativeContactFraction

contact_cutoff = 0.5  # nm

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
indices = np.unique(rna_id_contacts)
references = positions.astype("float")[np.asarray(indices)]
ncf = NativeContactFraction(indices, rna_id_contacts, references)
ncf_grad = grad(ncf.function)
grad_hot = ncf_grad(references + np.random.random(references.shape))
assert not np.any(np.isnan(grad_hot)), "force contains NaN values"
print("checking for forces of Q passed!")
