#!/usr/bin/env python3

import hoomd
import hoomd.dlext
import hoomd.md
import matplotlib.pyplot as plt
import numpy as np

import pysages
from pysages.backends import SamplingContext
from pysages.colvars import Component
from pysages.methods import HarmonicBias, HistogramLogger


def plot(xi_hist, target_hist, lim, name):
    fig, ax = plt.subplots()

    ax.set_xlabel(r"CV $\xi_i$")
    ax.set_ylabel(r"$p(\xi_i)$")

    x = np.linspace(lim[0], lim[1], xi_hist[0].shape[0])

    for i in range(len(xi_hist)):
        (line,) = ax.plot(x, xi_hist[i], label="i= {0}".format(i))
        ax.plot(x, target_hist[i], "--", color=line.get_color())

    ax.legend(loc="best")
    fig.savefig(str(name) + ".png")
    plt.close(fig)


def validate_hist(xi_hist, target, epsilon=0.1):
    assert len(xi_hist) == len(target)
    for i in range(len(xi_hist)):
        val = np.sqrt(np.mean((xi_hist[i] - target[i]) ** 2))
        if val > epsilon:
            raise RuntimeError(f"Biased histogram deviation too large: {val} epsilon {epsilon}")


def get_target_dist(center, k, lim, bins):
    x = np.linspace(lim[0], lim[1], bins)
    p = np.exp(-0.5 * k * (x - center) ** 2)
    # norm numerically
    p *= (lim[1] - lim[0]) / np.sum(p)
    return p


def generate_context(device=hoomd.device.CPU(), seed=0, gamma=1.0, **kwargs):
    sim = hoomd.Simulation(device=device, seed=seed)
    sim.create_state_from_gsd("start.gsd")
    integrator = hoomd.md.Integrator(dt=0.01)

    nl = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPD(nlist=nl, kT=1.0, default_r_cut=1.0)
    dpd.params[("A", "A")] = dict(A=kwargs.get("A", 5.0), gamma=gamma)
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

    k = 15
    method = HarmonicBias(cvs, k, center_cv)
    callback = HistogramLogger(100)

    # Parameters for plotting the histograms
    Lmax = 5.0
    bins = 25
    target_hist = []
    for i in range(len(center_cv)):
        target_hist.append(get_target_dist(center_cv[i], k, (-Lmax / 2, Lmax / 2), bins))
    lims = [(-Lmax / 2, Lmax / 2) for i in range(3)]

    # Running from a SamplingContext. This is only needed for restarting the
    # simulation within the same script/notebook and shown here as an example.
    # Generally, prefer starting without a SamplingContext, that is
    #
    # state = pysages.run(method, generate_context, timesteps, callback)
    #
    # instead of the two lines below
    sampling_context = SamplingContext(method, generate_context, callback)
    state = pysages.run(sampling_context, int(1e5))  # run a first time

    # Plot the histogram so far
    hist, edges = callback.get_histograms(bins=bins, range=lims)
    hist_list = [
        np.sum(hist, axis=(1, 2)) / (Lmax**2),
        np.sum(hist, axis=(0, 2)) / (Lmax**2),
        np.sum(hist, axis=(0, 1)) / (Lmax**2),
    ]
    plot(hist_list, target_hist, (-Lmax / 2, Lmax / 2), 1)
    validate_hist(hist_list, target_hist)

    # Run a second time within the same script when using a SamplingContext
    state = pysages.run(sampling_context, int(1e4))

    # Plot the histogram with the newly collected info
    hist, edges = callback.get_histograms(bins=bins, range=lims)
    hist_list = [
        np.sum(hist, axis=(1, 2)) / (Lmax**2),
        np.sum(hist, axis=(0, 2)) / (Lmax**2),
        np.sum(hist, axis=(0, 1)) / (Lmax**2),
    ]
    plot(hist_list, target_hist, (-Lmax / 2, Lmax / 2), 2)

    # Save the system's information to perform a restart in a new run.
    pysages.save(state, "restart.pkl")

    # Load the restart file. This is how to run a pysages run from a
    # previously stored state.
    state = pysages.load("restart.pkl")

    # When restarting, run the system using the same generate_context function!
    state = pysages.run(state, generate_context, int(1e4))

    # Plot all the accumulated data
    hist, edges = callback.get_histograms(bins=bins, range=lims)
    hist_list = [
        np.sum(hist, axis=(1, 2)) / (Lmax**2),
        np.sum(hist, axis=(0, 2)) / (Lmax**2),
        np.sum(hist, axis=(0, 1)) / (Lmax**2),
    ]
    plot(hist_list, target_hist, (-Lmax / 2, Lmax / 2), 3)


if __name__ == "__main__":
    main()
