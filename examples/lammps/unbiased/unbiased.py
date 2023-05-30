#!/usr/bin/env python3
import pdb
import lammps

from lammps import lammps
import lammps.dlext as lammps_dlext
import numpy as np

import pysages
from pysages.colvars import Component
from pysages.methods import HistogramLogger, Unbiased


def generate_context(**kwargs):
    # this list of arguments turns on the kokkos package with the GPU backend
    #   and add kk suffix to the supported styles
    args="-k on g 1 -sf kk -sc none -nocite"
    #args=["-k", "on", "g", "1", "-sf", "kk", "-sc", "none", "-nocite"]
    args=args.split()

    context = lammps(cmdargs=args)

    # set up a 3d Lennard-Jones melt
    context.command("units        lj")
    context.command("atom_style   atomic")
    context.command("lattice	  fcc 0.8442") 
    context.command("region	      box block 0 10 0 10 0 10") 
    context.command("create_box	  1 box") 
    context.command("create_atoms 1 box") 
    context.command("mass		  1 1.0") 
    context.command("velocity	  all create 1.0 182207 loop geom") 
    context.command("pair_style	  lj/cut 2.5") 
    context.command("pair_coeff	  1 1 1.0 1.0 2.5") 
    context.command("neighbor     0.3 bin") 
    context.command("neigh_modify delay 0 every 5 check yes") 
    context.command("fix          1 all nve") 

    # or run infile one line at a time
    #  infile=kwargs.get('infile')
    #  lines = open(infile,'r').readlines()
    #  for line in lines: context.command(line)

    return context


def main():
    cvs  = [Component([0], 2)]
    cvs += [Component([0], 1)]
    cvs += [Component([0], 0)]

    center_cv = [0.0]
    center_cv += [1.0, -0.3]

    method = Unbiased(cvs)
    callback = HistogramLogger(10)
    timesteps = 200

    # generate the simulation context
    context = generate_context()

    # create a LAMMPS view object
    view = lammps_dlext.LAMMPSView(context)
    N = view.global_particle_number()
    print(f"Total number of particles = {N}")
    cuda_enabled = view.has_kokkos_cuda_enabled()
    print(f"CUDA enabled = {cuda_enabled}")

    # create a Sampler object
    args = "mydlext all dlext"
    args=args.split()
    # TODO: this still crashes
    #sampler = lammps_dlext.FixDLExt(context, args)

    # attempt to get the atom positions
    #x = sampler.positions()

    #raw_result = pysages.run(method, generate_context, timesteps, callback)
    #print(np.asarray(raw_result.callbacks[0].data))

    # closing LAMMPS
    context.close()
    context.finalize()


if __name__ == "__main__":
    main()
