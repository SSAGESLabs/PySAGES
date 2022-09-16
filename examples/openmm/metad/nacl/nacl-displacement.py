#!/usr/bin/env python3

"""
Metadynamics simulation of NaCl in water with OpenMM and PySAGES.

Example command to run the simulation `python3 nacl-displacement.py --time-steps 1000`
For other supported commandline parameters, check `python3 nacl-displacement.py --help`

Additional optional dependencies:
 - [openmmforcefields](https://github.com/openmm/openmmforcefields)
"""

# %%
import sys
import time

from nacl import generate_simulation, get_args, kB

import pysages
from pysages.colvars import Displacement
from pysages.methods import MetaDLogger, Metadynamics


# %%
def main(argv=[]):
    args = get_args(argv)

    cvs = [Displacement([509, 510])]

    height = 1.2  # kJ/mol
    sigma = [0.05]  # nm
    deltaT = 5000 if args.well_tempered else None
    stride = 500  # frequency for depositing gaussians
    timesteps = args.time_steps
    ngauss = timesteps // stride + 1  # total number of gaussians

    # 1D Grid for storing bias potential and its gradient
    grid = pysages.Grid(lower=(0.5, 0.5, 0.5), upper=(2.5, 2.5, 2.5), shape=(50, 50, 50))
    grid = grid if args.use_grids else None

    # Method
    method = Metadynamics(cvs, height, sigma, stride, ngauss, deltaT=deltaT, kB=kB, grid=None)

    # Logging
    hills_file = "hills.dat"
    callback = MetaDLogger(hills_file, stride) if args.log else None

    tic = time.perf_counter()
    run_result = pysages.run(method, generate_simulation, timesteps, callback)
    toc = time.perf_counter()
    print(f"Completed the simulation in {toc - tic:0.4f} seconds.")

    return run_result


# %%
if __name__ == "__main__":
    main(sys.argv[1:])
