#!/usr/bin/env python

import barnaba as bb
import numpy as np

native = "../../inputs/GAGA.box_0mM.pdb"
traj = "output.dcd"
# traj = native  #'output.dcd'
ermsd = bb.ermsd(native, traj, topology=native, cutoff=3.2)
np.savetxt("ermsd_barnaba.txt", ermsd)
