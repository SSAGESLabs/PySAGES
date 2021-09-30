#!/usr/bin/python3
import sys

import gsd
import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt

def get_positions(traj):
    pos = np.zeros((len(traj), traj[0].particles.N, 3))
    for (i, snap) in zip(range(len(traj)), traj):
        pos[i] = snap.particles.position
    return pos


def get_histogram(pos, idx, L, N=25):
    hist = np.zeros((3, N))
    for i in range(3):
        hist[i], _ = np.histogram(pos[:,idx, i], bins=N, range=(-L[i]/2, L[i]/2), density=True)
    return hist

def plot(unbiased_hist, biased_hist):
    fig, ax = plt.subplots()

    ax.set_xlabel(r"coordinate $X_\alpha/L$")
    ax.set_ylabel(r"$p(X_\alpha)$")

    x = np.linspace(0, 1, unbiased_hist.shape[1])
    ax.plot(x, unbiased_hist[0], label="unbiased X")
    ax.plot(x, unbiased_hist[1], label="unbiased Y")
    ax.plot(x, unbiased_hist[2], label="unbiased Z")

    ax.plot(x, biased_hist[0], label="biased X")
    ax.plot(x, biased_hist[1], label="biased Y")
    ax.plot(x, biased_hist[2], label="biased Z")

    ax.legend(loc="best")
    fig.savefig("hist.pdf")
    plt.close(fig)


def main(argv):
    if len(argv) != 1:
        print("Usage: ./analyze_gsd.py filename")
        return
    filename = argv[0]

    with gsd.hoomd.open(filename, "rb") as traj:
        pos = get_positions(traj)
        L = traj[0].configuration.box[:3]
    biased_hist = get_histogram(pos, 0, L)
    unbiased_hist = get_histogram(pos, 1, L)

    plot(unbiased_hist, biased_hist)


if __name__ == "__main__":
    main(sys.argv[1:])
