#!/usr/bin/env python3

import hoomd
import hoomd.dlext
import hoomd.md
import numpy as np

import pysages
from pysages.colvars import Component
from pysages.methods import HistogramLogger, Unbiased


def generate_context(**kwargs):
    sim = hoomd.Simulation(
        device=kwargs.get("context", hoomd.device.CPU()), seed=kwargs.get("seed", 1)
    )
    sim.create_state_from_gsd("start.gsd")
    integrator = hoomd.md.Integrator(dt=0.01)

    nl = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPD(nlist=nl, kT=1.0, default_r_cut=1.0)
    dpd.params[("A", "A")] = dict(A=kwargs.get("A", 5.0), gamma=kwargs.get("gamma", 1.0))
    integrator.forces.append(dpd)
    nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
    integrator.methods.append(nve)
    sim.operations.integrator = integrator
    return sim


def main():
    cvs = [Component([0], 2)]
    cvs += [Component([0], 1)]
    cvs += [Component([0], 0)]

    center_cv = [0.0]
    center_cv += [1.0, -0.3]

    method = Unbiased(cvs)
    callback = HistogramLogger(10)

    raw_result = pysages.run(method, generate_context, int(1e2), callback)

    print(np.asarray(raw_result.callbacks[0].data))


if __name__ == "__main__":
    main()
