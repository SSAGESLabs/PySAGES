#!/usr/bin/env python3
import pdb
import lammps
#import lammps.dlext
from lammps import lammps
import numpy as np

import pysages
from pysages.colvars import Component
from pysages.methods import HistogramLogger, Unbiased


def generate_context(**kwargs):

    infile = kwargs[1]
    args="-k on g 1 -sf kk"
    args=args.split()
    #args=["-k", "on", "g", "1", "-sf", "kk"]
    context = lammps(cmdargs=args)

    # run infile one line at a time
    lines = open(infile,'r').readlines()
    for line in lines: context.command(line)
    return context


def main():
    cvs = [Component([0], 2)]
    cvs += [Component([0], 1)]
    cvs += [Component([0], 0)]

    center_cv = [0.0]
    center_cv += [1.0, -0.3]

    method = Unbiased(cvs)
    callback = HistogramLogger(10)

    timesteps = 200
    raw_result = pysages.run(method, generate_context, timesteps, callback)
    print(np.asarray(raw_result.callbacks[0].data))


if __name__ == "__main__":
    main()
