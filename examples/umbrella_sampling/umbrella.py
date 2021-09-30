#!/usr/bin/env python3

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

        cvs = [Component([0], 2),]
        # cvs += [Component([0], 1),]
        # cvs += [Component([0], 0),]

        center_cv = [ 0.,]
        # center_cv += [1, -1]

        method = UmbrellaSampling(cvs, 10., center_cv)

        hoomd.md.integrate.nve(group=hoomd.group.all())
        hoomd.md.integrate.mode_standard(dt=0.01)

        nl = hoomd.md.nlist.cell()
        dpd = hoomd.md.pair.dpd(r_cut=1,nlist=nl,seed=42,kT=1.)
        dpd.pair_coeff.set("A","A",A=5.,gamma=1.0)

        hoomd.analyze.log("umbrella.dat", qr, 100, overwrite=True)

        bind(context, method)

        gsd = hoomd.dump.gsd("umbrella.gsd", group=hoomd.group.all(), period=100, overwrite=True)

        hoomd.run(1e5, limit_multiple=100)



if __name__ == "__main__":
    main()
