#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import hoomd
import hoomd.md as md
import hoomd.dlext

import pysages
from pysages.collective_variables import Component
from pysages.methods import UmbrellaIntegration


param1 = {"A": 0.5, "w": 0.2, "p": 2}

def generate_context(**kwargs):
    hoomd.context.initialize("")
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

        periodic = hoomd.md.external.periodic()
        periodic.force_coeff.set('A', A=param1["A"], i=0, w=param1["w"], p=param1["p"])
        periodic.force_coeff.set('B', A=0.0, i=0, w=0.02, p=1)
    return context


def plot_hist(result, bins=50):
    fig, ax = plt.subplots(2, 2)

    # ax.set_xlabel("CV")
    # ax.set_ylabel("p(CV)")

    counter = 0
    hist_per = len(result["center"])//4+1
    for x in range(2):
        for y in range(2):
            for i in range(hist_per):
                if counter+i < len(result["center"]):
                    center = np.asarray(result["center"][counter+i])
                    histo, edges = result["histogram"][counter+i].get_histograms(bins=bins)
                    edges = np.asarray(edges)[0]
                    edges = (edges[1:] + edges[:-1]) / 2
                    ax[x,y].plot(edges, histo, label="center {0}".format(center))
                    ax[x,y].legend(loc="best", fontsize="xx-small")
                    ax[x,y].set_yscale("log")
            counter += hist_per
    while counter < len(result["center"]):
        center = np.asarray(result["center"][counter])
        histo, edges = result["histogram"][counter].get_histograms(bins=bins)
        edges = np.asarray(edges)[0]
        edges = (edges[1:] + edges[:-1]) / 2
        ax[1,1].plot(edges, histo, label="center {0}".format(center))
        counter += 1

    fig.savefig("hist.pdf")

def external_field(r, A, p, w):
    return A*np.tanh(1/(2*np.pi*p*w)*np.cos(p*r))

def plot_energy(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Free energy $[\epsilon]$")
    center = np.asarray(result["center"])
    a_free_energy = np.asarray(result["a_free_energy"])
    offset = np.min(a_free_energy)
    ax.plot(center, a_free_energy-offset, color="teal")

    x = np.linspace(-3, 3, 50)
    data = external_field(x, **param1)
    offset = np.min(data)
    ax.plot(x, data-offset, label="test")

    fig.savefig("energy.pdf")


def get_args(argv):
    parser = argparse.ArgumentParser(description="Example script to run umbrella integration")
    parser.add_argument("--k-spring", "-k", type=float, default=50., help="spring constant for each replica")
    parser.add_argument("--N-replica", "-N", type=int, default=25, help="Number of replica along the path")
    parser.add_argument("--start-path", "-s", type=float, default=-1.5, help="Start point of the path")
    parser.add_argument("--end-path", "-e", type=float, default=1.5, help="Start point of the path")
    parser.add_argument("--time-steps", "-t", type=int, default=int(1e5), help="Number of simulation steps for each replica")
    parser.add_argument("--log-period", "-l", type=int, default=int(50), help="Frequency of logging the CVS for histogram")
    parser.add_argument("--discard-equi", "-d", type=int, default=int(1e4), help="Discard timesteps before logging for equilibration")
    args = parser.parse_args(argv)
    return args

def main(argv):

    args = get_args(argv)

    cvs = [Component([0], 0),]
    method = UmbrellaIntegration(cvs)

    centers = list(np.linspace(args.start_path, args.end_path, args.N_replica))
    result = method.run(generate_context, args.time_steps, centers, args.k_spring, args.log_period, args.discard_equi)

    plot_energy(result)
    plot_hist(result)


if __name__ == "__main__":
    main(sys.argv[1:])
