from jax import numpy as np
import numpy as onp

import matplotlib
import matplotlib.pyplot as plt


def format_plot(x, y):
    plt.xlabel(x, fontsize=20)
    plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1],
    )
    plt.tight_layout()


def plot_system(R, box_size, species=None, ms=20):
    R_plt = onp.array(R)

    if species is None:
        plt.plot(R_plt[:, 0], R_plt[:, 1], "o", markersize=ms)
    else:
        for ii in range(np.amax(species) + 1):
            Rtemp = R_plt[species == ii]
            plt.plot(Rtemp[:, 0], Rtemp[:, 1], "o", markersize=ms)

    plt.xlim([0, box_size])
    plt.ylim([0, box_size])
    plt.xticks([], [])
    plt.yticks([], [])

    finalize_plot((1, 1))
