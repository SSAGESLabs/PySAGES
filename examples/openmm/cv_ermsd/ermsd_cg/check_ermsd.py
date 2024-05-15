#!/usr/bin/env python
# the ermsd calculation code is taken from barnaba package
# github.com/srnas/barnaba

import numpy as np
from scipy.spatial import distance

ermsd_cg = np.loadtxt("ermsd_cg.txt")
ermsd_aa = np.loadtxt("ermsd.txt")

assert (
    np.mean((ermsd_cg - ermsd_aa) ** 2) < 1e-3
), "the difference between pysages ermsd and barnaba version is too large!"

print("checking for rmsd passed!")
