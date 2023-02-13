###!/usr/bin/env python -i
# preceding line should have path for Python on your machine

# simple.py
# Serial syntax: simple.py in.lj
#                in.lj = LAMMPS input script

# Parallel syntax: mpirun -np 4 simple.py in.lj
#                  in.lj = LAMMPS input script
# also need to uncomment mpi4py sections below

#
# LAMMPS build with the following configuration:
# ccmake -S cmake -B build -D PKG_KOKKOS=on -D Kokkos_ENABLE_CUDA=on -D Kokkos_ARCH_AMPERE80=on \
#     -D PKG_MOLECULE=on -D PKG_KSPACE=on -D BUILD_SHARED_LIBS=on -D PKG_PYTHON=on -D FFT=KISS \
#     -D CMAKE_INSTALL_PREFIX=`python3 -c "import sys; print(sys.prefix)"` -D Python_EXECUTABLE=`which python3`
#
# If the build succeeds, the shared library liblammps.so is installed into $CMAKE_INSTALL_PREFIX/lib64.
# Also, we also need the shared lib libpython3.x.so.1, which is $CMAKE_INSTALL_PREFIX/lib. Depending on the miniconda/anaconda version
# these two paths may, or may not, be prepended to LD_LIBRARY_PATH. If they are not, do so
#
# export LD_LIBRARY_PATH=$CMAKE_INSTALL_PREFIX/lib64/lib:$CMAKE_INSTALL_PREFIX/lib64/lib64:$LD_LIBRARY_PATH
#
# where $CMAKE_INSTALL_PREFIX is the full path to the top-level folder of the virtual environment.
#
# To test the installation of the LAMMPS python module
#      python3 -c "from lammps import lammps; lmp = lammps()"
#

from __future__ import print_function
import sys

# parse command line

argv = sys.argv
if len(argv) != 2:
  print("Syntax: simple.py in.lj")
  sys.exit()

infile = sys.argv[1]

me = 0

# uncomment this if running in parallel via mpi4py
#from mpi4py import MPI
#me = MPI.COMM_WORLD.Get_rank()
#nprocs = MPI.COMM_WORLD.Get_size()

import lammps
from lammps import lammps
#print(lammps.__file__)
#args="-k on g 1 -sf kk"
args="-nocite"
args=args.split()
#args=["-k", "on", "g", "1", "-sf", "kk"]
lmp = lammps(cmdargs=args)

# run infile one line at a time

lines = open(infile,'r').readlines()
for line in lines: lmp.command(line)

# run 10 more steps

lmp.command("run 10")
