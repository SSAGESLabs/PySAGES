#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import hoomd
hoomd.context.initialize()
import hoomd.md as md
import hoomd.dlext

import pysages
from pysages.collective_variables import Component
from pysages.methods import UmbrellaSampling
from pysages.runners import run_simple

class HistogramLogger:
    def __init__(self, period):
        self.period = period
        self.data = []

    def __call__(self, snapshot, state, timestep):
        if timestep % self.period == 0:
            self.data.append(state.xi)

    def get_histograms(self, bins, lim):
        data = np.asarray(self.data)
        data = data.reshape(data.shape[0], data.shape[2])
        histograms = []
        for i in range(data.shape[1]):
            histograms.append(np.histogram(data[:,i], bins=bins, range=lim, density=True)[0])
        return histograms


def plot(xi_hist, target_hist, lim):
    fig, ax = plt.subplots()

    ax.set_xlabel(r"CV $\xi_i$")
    ax.set_ylabel(r"$p(\xi_i)$")

    x = np.linspace(lim[0], lim[1], xi_hist[0].shape[0])

    for i in range(len(xi_hist)):
        line, = ax.plot(x, xi_hist[i], label="i= {0}".format(i))
        ax.plot(x, target_hist[i], "--", color=line.get_color())

    ax.legend(loc="best")
    fig.savefig("hist.pdf")
    plt.close(fig)


def validate_hist(xi_hist, target, epsilon=0.1):
    assert len(xi_hist) == len(target)
    for i in range(len(xi_hist)):
        val = np.sqrt(np.mean((xi_hist[i]-target[i])**2))
        if val > epsilon:
            raise RuntimeError("Biased historgram deviation too large: {0} epsilon {1}".format(val, epsilon))


def get_target_dist(center, k, lim, bins):
    x = np.linspace(lim[0], lim[1], bins)
    p = np.exp(-0.5*k*(x-center)**2)
    # norm numerically
    p *= (lim[1]-lim[0])/np.sum(p)
    return p


def generate_context(**kwargs):
    context = hoomd.context.SimulationContext()
    with context:
        system = hoomd.init.read_gsd("start.gsd")

        hoomd.md.integrate.nve(group=hoomd.group.all())
        hoomd.md.integrate.mode_standard(dt=0.01)

        nl = hoomd.md.nlist.cell()
        dpd = hoomd.md.pair.dpd(r_cut=1,nlist=nl,seed=42,kT=1.)
        dpd.pair_coeff.set("A","A",A=kwargs.get("A", 5.),gamma=kwargs.get("gamma", 1.))
    return context


def main():

    cvs = [Component([0], 2),]
    cvs += [Component([0], 1),]
    cvs += [Component([0], 0),]

    center_cv = [ 0.,]
    center_cv += [0.3, -0.3]

    k= 15
    method = UmbrellaSampling(cvs, k, center_cv)
    callback = HistogramLogger(100)

    run_simple(generate_context, method, int(1e5), callback, {"A":7.}, profile=True)

    # Lmax = np.max([system.box.Lx, system.box.Ly, system.box.Lz])
    Lmax = 5.
    bins = 25
    target_hist = []
    for i in range(len(center_cv)):
        target_hist.append(get_target_dist(center_cv[i], k, (-Lmax/2, Lmax/2), bins))
    hist = callback.get_histograms(bins, (-Lmax/2, Lmax/2))
    plot(hist, target_hist, (-Lmax/2, Lmax/2))
    validate_hist(hist, target_hist)


if __name__ == "__main__":
    main()
