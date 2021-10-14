#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import hoomd
hoomd.context.initialize()
import hoomd.md as md
import hoomd.dlext

import pysages
from pysages.collective_variables import Component
from pysages.methods import UmbrellaIntegration

def generate_context(**kwargs):
    context = hoomd.context.SimulationContext()
    with context:
        print("Operating replica {0}".format(kwargs.get("replica_num")))
        system = hoomd.init.read_gsd("start.gsd")

        hoomd.md.integrate.nve(group=hoomd.group.all())
        hoomd.md.integrate.mode_standard(dt=0.01)

        nl = hoomd.md.nlist.cell()
        dpd = hoomd.md.pair.dpd(r_cut=1, nlist=nl, seed=42, kT=1.)
        dpd.pair_coeff.set("A", "A", A=5., gamma=1.0)
        dpd.pair_coeff.set("A", "B", A=5., gamma=1.0)
        dpd.pair_coeff.set("B", "B", A=5., gamma=1.0)

        amplitude = 1.
        periodic = hoomd.md.external.periodic()
        periodic.force_coeff.set('A', A=amplitude, i=0, w=0.02, p=2)
        periodic.force_coeff.set('B', A=0.0, i=0, w=0.02, p=1)
    return context


def plot(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("p(CV)")


    for i in range(len(result["center"])):
        center = np.asarray(result["center"][i])
        edges = np.asarray(result["histogram_edges"][i][0])
        histo = np.asarray(result["histogram"][i])
        print(edges.shape)
        print(histo.shape)
        ax.plot(edges[1:], histo, label="center {0}".format(center))
    ax.legend(loc="best")

    ax2 = ax.twinx()
    ax2.set_ylabel("Free energy")
    center = np.asarray(result["center"])
    A = np.asarray(result["A"])
    ax2.plot(center, A, color="black")

    fig.savefig("hist.pdf")


def main():
    cvs = [Component([0], 0),]
    method = UmbrellaIntegration(cvs)

    k = 15.
    centers = list(np.linspace(-0.3, 0.3, 25))
    result = method.run(generate_context, int(1e5), centers, k, 10, 35, (-0.5, 0.5))
    # for key in result.keys():
    #     print(key, result[key])
    plot(result)


if __name__ == "__main__":
    main()
