#!/usr/bin/env python
# the ermsd calculation code is taken from barnaba package
# github.com/srnas/barnaba

import numpy as np
from openmm import app, unit
from scipy.spatial import distance as sd


def calc_q(contact_dists, contact_dists0, gamma=50, lambda_d=1.5, clip=False, clip_val=5):
    N_contacts = len(contact_dists)
    diff = gamma * (contact_dists - lambda_d * contact_dists0)
    if clip:
        diff = np.clip(diff, None, clip_val)

    Q = 1 / N_contacts * np.sum(1 / (1 + np.exp(diff)))

    return Q


def pos2q(pos, contact_pairs, contact_dists0, gamma=50, lambda_d=1.5, clip=False, clip_val=5):
    dist_matrix = sd.squareform(sd.pdist(pos))
    contact_dists = dist_matrix[contact_pairs[:, 0], contact_pairs[:, 1]]
    Q = calc_q(
        contact_dists, contact_dists0, gamma=gamma, lambda_d=lambda_d, clip=clip, clip_val=clip_val
    )

    return Q


contact_pairs = np.load("contact_pairs.npy", allow_pickle=True)
contact_pairs_remapped = np.load("contact_pairs_remapped.npy", allow_pickle=True)
references = np.loadtxt("references.txt")
dist_matrix_ref = sd.squareform(sd.pdist(references))
contact_dists0 = dist_matrix_ref[contact_pairs_remapped[:, 0], contact_pairs_remapped[:, 1]]

traj = app.PDBFile("output.pdb")
n_frames = traj.getNumFrames()
Q_posthoc = []
for i in range(n_frames):
    pos = traj.getPositions(asNumpy=True, frame=i).value_in_unit(unit.nanometer).astype("float")
    Q_hot = pos2q(pos, contact_pairs, contact_dists0, gamma=50, lambda_d=1.5, clip=True)
    Q_posthoc.append(Q_hot)

Q_pysages = np.loadtxt("Q.txt")
np.savetxt("Q_posthoc.txt", Q_posthoc)

assert (
    np.mean((Q_pysages - Q_posthoc) ** 2) < 1e-6
), "the difference between pysages Q and post-hoc calculation is too large!"

print("checking for Q passed!")
