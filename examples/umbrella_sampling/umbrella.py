#!/usr/bin/env python3

import h5py
import numpy as np

import hoomd
hoomd.context.initialize()
import hoomd.md as md
import hoomd.dlext

import pysages
from pysages.ssages.collective_variables import Component
from pysages.ssages.methods import UmbrellaSampling
from pysages.backends import bind

def main():
    with hoomd.context.SimulationContext() as context:
        system = hoomd.init.read_gsd("start.gsd")

        #gather basic lognames
        qr = []
        #log some thermo properties
        qr += ['temperature', 'potential_energy', 'kinetic_energy']

        cvs = (Component([0], 0), Component([0], 1))

        method = UmbrellaSampling(cvs, 10., (-2, 2))

        hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=0.1)
        hoomd.md.integrate.mode_standard(dt=0.05)

        bind(context, method)

        gsd = hoomd.dump.gsd("umbrella.gsd", group=hoomd.group.all(), period=100, overwrite=True)

        hoomd.run(1e6, limit_multiple=100)



if __name__ == "__main__":
    main()
