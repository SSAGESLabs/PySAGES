#!/usr/bin/env python3

import sys
import jax
import jax.numpy as np
from pysages.ssages.methods.string_method import interpolate_cv
from mpi4py import MPI
import matplotlib.pyplot as plt


def create_circle(radius, size):
    image = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            if (x-size/2)**2 + (y-size/2)**2 < radius**2:
                image = jax.ops.index_add(image, (x, y), 1)
    return image


def create_slab(width, size):
    image = np.zeros((size, size))
    for x in range(size):
        if np.abs(x-size/2) < width/2:
            image = jax.ops.index_add(image, jax.ops.index[x, :], 1)
    return image


def plot_image(image, name):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="plasma", vmin=0, vmax=1)
    fig.savefig(name, transparent=True, bbox_inches="tight")
    plt.close(fig)


def main(argv):
    if len(argv) != 0:
        print("Usage: ./string.py")
        return

    comm = MPI.COMM_WORLD.Dup()
    size = 50
    radius = 15
    width = np.pi*radius**2/size

    image = None
    if comm.Get_rank() == 0:
        image = create_circle(radius, size)
        plot_image(image, "start0.png")

    if comm.Get_rank() == comm.Get_size()-1:
        image = create_slab(width, size)
        plot_image(image, "start1.png")

    image = interpolate_cv(comm.Get_rank()/(comm.Get_size()-1), comm, image)
    plot_image(image, "end{0}.png".format(comm.Get_rank()))

if __name__ == "__main__":
    main(sys.argv[1:])
