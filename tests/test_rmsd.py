#!/usr/bin/env python

import numpy as np

from pysages.colvars.orientation import RMSD_Kabsch

base = np.loadtxt("tests/ci2_1.txt", dtype=str)[:, 6:9].astype(float)
reference = np.loadtxt("tests/ci2_2.txt", dtype=str)[:, 6:9].astype(float)
indices = np.arange(len(base))
rmsd_kabsch = RMSD_Kabsch(indices, reference)
rmsd_calculated = rmsd_kabsch.function(base)
rmsd_expected = 11.7768
if not np.allclose(rmsd_expected, rmsd_calculated):
    raise ValueError(f"Test Case PDB 1 failed: {rmsd_expected, rmsd_calculated}")

base = np.loadtxt("tests/ci2_1t.txt", dtype=str)[:, 6:9].astype(float)
reference = np.loadtxt("tests/ci2_1.txt", dtype=str)[:, 6:9].astype(float)
rmsd_kabsch = RMSD_Kabsch(indices, reference)
rmsd_calculated = rmsd_kabsch.function(base)
rmsd_expected = 0.0
if not np.allclose(rmsd_expected, rmsd_calculated, atol=1e-3):
    raise ValueError(f"Test Case PDB 2 failed: {rmsd_expected, rmsd_calculated}")


base = np.loadtxt("tests/ci2_2.txt", dtype=str)[:, 6:9].astype(float)
reference = np.loadtxt("tests/ci2_12.txt", dtype=str)[:, 6:9].astype(float)
rmsd_kabsch = RMSD_Kabsch(indices, reference)
rmsd_calculated = rmsd_kabsch.function(base)
rmsd_expected = 15.422431
if not np.allclose(rmsd_expected, rmsd_calculated):
    raise ValueError(f"Test Case PDB 3 failed: {rmsd_expected, rmsd_calculated}")
