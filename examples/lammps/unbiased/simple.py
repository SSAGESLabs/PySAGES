#!/usr/bin/env python3
# preceding line should have path for Python on your machine
#
# simple.py
# Serial syntax: ./simple.py in.lj
#                in.lj = LAMMPS input script
#
# LAMMPS build with the following configuration to enable the KOKKOS and PYTHON packages:
# ccmake -S cmake -B build -D PKG_KOKKOS=on -D Kokkos_ENABLE_CUDA=on -D Kokkos_ARCH_AMPERE80=on \
#     -D PKG_MOLECULE=on -D PKG_KSPACE=on -D BUILD_SHARED_LIBS=on -D PKG_PYTHON=on -D FFT=KISS \
#     -D CMAKE_INSTALL_PREFIX=`python3 -c "import sys; print(sys.prefix)"` -D Python_EXECUTABLE=`which python3`
#
# If the build succeeds, the shared library liblammps.so is installed into $CMAKE_INSTALL_PREFIX/lib64.
# We also need the shared lib libpython3.x.so.1, which is under $CMAKE_INSTALL_PREFIX/lib.
# Finally, libdlext.so installed by lammps.dlext is currently not found, so we need to add
# $CMAKE_INSTALL_PREFIX/lib/python3.9/site-packages/lammps to LD_LIBRARY_PATH.
#
# export LD_LIBRARY_PATH=$CMAKE_INSTALL_PREFIX/lib64/lib:$CMAKE_INSTALL_PREFIX/lib64:$CMAKE_INSTALL_PREFIX/lib/python3.9/site-packages/lammps:$LD_LIBRARY_PATH
#
# where $CMAKE_INSTALL_PREFIX is the full path to the top-level folder of the virtual environment.
#
# To test the installation of the LAMMPS python module
#      python3 -c "from lammps import lammps; lmp = lammps()"
#

from __future__ import print_function
import sys

import lammps
from lammps import lammps

def main():

    # parse command line   
    argv = sys.argv
    if len(argv) != 2:
      print("Syntax: ./simple.py in.lj")
      sys.exit()

    infile = sys.argv[1]

    args = "-k on g 1 -sf kk"
    #args += " -nocite -sc none"
    args=args.split()
    lmp = lammps(cmdargs=args)

    # run infile one line at a time to set up the simulation
    lines = open(infile,'r').readlines()
    for line in lines: lmp.command(line)

    # this is wrapped as the run method, see methods_dispatch run()
    #   wrapped_context.run(timesteps, **kwargs) is called in methods/core.py
    # do "run 0" to setup neighbor build and force computes
    lmp.command("run 0")

    # demo of how to extract simulation information from lmp (context)"
    # see LMP_SRC/python/lammps/core.py, or under the installation path
    #  $CMAKE_INSTALL_PREFIX/lib/python3.9/site-packages/lammps/core.py

    print("----------------------------------------------")
    print("Simulation info from the LAMMPS python module:")
    natoms = lmp.extract_global("natoms")
    print(f"natoms = {natoms}")
    nlocal = lmp.extract_setting('nlocal')
    print(f"nlocal = {nlocal}")

    boxlo,boxhi,xy,yz,xz,periodicity,_ = lmp.extract_box()
    print(f"box lo = {boxlo}")
    print(f"box hi = {boxhi}")
    Lx = boxhi[0] - boxlo[0]
    Ly = boxhi[1] - boxlo[1]
    Lz = boxhi[2] - boxlo[2]
    origin = boxlo
    H = ((Lx, xy * Ly, xz * Lz), (0.0, Ly, yz * Lz), (0.0, 0.0, Lz))
    print("box matrix: ", H)

    # or
    #boxlo = lmp.extract_global("boxlo")
    #boxhi = lmp.extract_global("boxhi")

    dim = lmp.extract_setting('dimension')
    print("Dimensionality = ", dim)
    pe = lmp.get_thermo("pe")
    print(f"Current potential energy = {pe}")

    dt = lmp.extract_global("dt")
    print(f"Timestep dt = {dt}")

    # Query if PKG_KOKKOS is enabled
    pkg_kokkos_enabled = lmp.has_package('KOKKOS')
    print(f"KOKKOS enabled = {pkg_kokkos_enabled}")

    on_gpu = False
    kokkos_backends = {}
    if pkg_kokkos_enabled == True:
        kokkos_backends = lmp.accelerator_config["KOKKOS"]
        if 'cuda' in kokkos_backends["api"]:
          on_gpu = True
    if on_gpu:
      print('  - KOKKOS with CUDA backend')
    else:
      print(kokkos_backends["api"])

    print("----------------------------------------------")

    # this is necessary to shut down the MPI communication when Kokkos environment is active
    lmp.finalize()

if __name__ == "__main__":
    main()