#!/usr/bin/env python3

"""
Metadynamics simulation of 2 particles in LJ fluid in 2D with JAX MD and PySAGES.

Example command to run the simulation `python3 particles-ljfluid.py --time-steps 1000`
For other supported commandline parameters, check `python3 particles-ljfluid.py --help`
"""

# %%
import argparse
import os
import sys
import time

import numpy
import pysages

from pysages.colvars import Distance
from pysages.methods import Metadynamics, MetaDLogger
from pysages.utils import try_import
from pysages.approxfun import compute_mesh

import jax.numpy as np
from jax import random, jit, lax, ops, grad
from jax.config import config

config.update("jax_enable_x64", True)

from jax_md import space, smap, energy, minimize, quantity, simulate
from dataclasses import dataclass
from utils import format_plot, finalize_plot, plot_system

from pysages.backends.core import JaxMDContext

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style(style="white")

# %%
# Define system parameters
N = 400  # number of particles
dimension = 2  # number of dimensions
density = 0.84  # density of the system
box_size = quantity.box_size_at_number_density(N, density, dimension)
dt = 5e-3  # time step
displacement, shift = space.periodic(box_size)
kT = 0.8  # temperature
particles_index = [0, 10]
kB = 1

# set epsilon to zero to exclude LJ interactions between particles
lj_epsilon = np.ones((N, N))
lj_epsilon = lj_epsilon.at[particles_index[0], particles_index[1]].set(0.0)
lj_epsilon = lj_epsilon.at[particles_index[1], particles_index[0]].set(0.0)

# set epsilon to zero to exclude LJ interactions between particles
morse_epsilon = np.zeros((N, N))
morse_epsilon = morse_epsilon.at[particles_index[0], particles_index[1]].set(5.0)
morse_epsilon = morse_epsilon.at[particles_index[1], particles_index[0]].set(5.0)

# %%
def shift_fn(R, dR, **kwargs):

    bias = kwargs.get("bias")

    return shift(R, dR) + bias


# %%
# Define energy function for 2-particles interacting via Morse
# potential in a LJ fluid
def total_energy_fn(R, **args):

    solvent_energy_fn = energy.lennard_jones_pair(
        displacement, sigma=1.0, epsilon=lj_epsilon, r_cutoff=2.5
    )

    particle_energy_fn = energy.morse_pair(
        displacement, sigma=1.0, epsilon=morse_epsilon, alpha=5.0, r_cutoff=2.5
    )

    return particle_energy_fn(R) + solvent_energy_fn(R)


# %%
# Define function to perform energy minimization
def run_minimization(energy_fn, R_init, shift, num_steps=5000):

    dt_start = 0.001
    dt_max = 0.004
    init, apply = minimize.fire_descent(jit(energy_fn), shift, dt_start=dt_start, dt_max=dt_max)
    apply = jit(apply)

    @jit
    def scan_fn(state, i):
        return apply(state), 0.0

    state = init(R_init)
    state, _ = lax.scan(scan_fn, state, np.arange(num_steps))

    return state.position, np.amax(np.abs(-grad(total_energy_fn)(state.position)))


# %%
def generate_simulation(total_energy_fn, shift_fn, dt, box, kT, state, log):

    # Log information about the simulation.
    # NOTE: Warning: As of v0.2.0 the internal representation of JAX MD simulations
    # has changed from velocity to momentum. For the most part this should not affect your work,
    # but the inputs to quantity.kinetic_energy and quantity.temperature have changed
    # from taking velocity to momentum!

    # NOTE: v0.2.0 is not released officially yet!

    # T = quantity.temperature(velocity=state.velocity)  # this line changes in v0.2.0

    # log["kT"] = log["kT"].at[i].set(T)
    # H = simulate.nvt_nose_hoover_invariant(total_energy_fn, state, kT)
    # log["H"] = log["H"].at[i].set(H)
    #
    ## Record positions every `write_every` steps.
    # log["position"] = lax.cond(
    #    i % write_every == 0,
    #    lambda p: p.at[i // write_every].set(state.position),
    #    lambda p: p,
    #    log["position"],
    # )
    #
    # print("log")
    # print(log)
    init_fn, apply = simulate.nvt_nose_hoover(total_energy_fn, shift_fn, dt, kT)
    force_fn = quantity.canonicalize_force(total_energy_fn(state))

    return JaxMDContext(box, force_fn, init_fn, apply, dt)


# %%
def get_args(argv):
    available_args = [
        ("well-tempered", "w", bool, 0, "Whether to use well-tempered metadynamics"),
        ("use-grids", "g", bool, 0, "Whether to use grid acceleration"),
        ("log", "l", bool, 0, "Whether to use a callback to log data into a file"),
        ("time-steps", "t", int, 5e5, "Number of simulation steps"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run metadynamics")
    for (name, short, T, val, doc) in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)
    return parser.parse_args(argv)


# %%
def main(argv=[]):
    args = get_args(argv)

    # Generate initial positions
    key = random.PRNGKey(0)
    key, split = random.split(key)
    R = box_size * random.uniform(split, (N, dimension), dtype=np.float64)
    box = np.repeat(box_size, dimension)

    # RUN energy minimization
    Rfinal, max_force_component = run_minimization(total_energy_fn, R, shift)
    print("largest component of force after minimization = {}".format(max_force_component))

    # Set up for NVT simulation
    # state = Rfinal
    # init_fn, apply = simulate.nvt_nose_hoover(total_energy_fn, shift, dt, kT)

    # Enhanced sampling with metadynamics
    cvs = [Distance(particles_index)]

    height = 1.2  # kT
    sigma = [0.5]  # sigma
    deltaT = 20 if args.well_tempered else None
    stride = 500  # frequency for depositing gaussians
    timesteps = args.time_steps
    ngauss = timesteps // stride + 1  # total number of gaussians
    write_every = 100

    # Grid for storing bias potential and its gradient
    grid = pysages.Grid(lower=(0), upper=(box_size * np.sqrt(2)), shape=(50), periodic=True)
    grid = grid if args.use_grids else None

    # Method
    method = Metadynamics(cvs, height, sigma, stride, ngauss, deltaT=deltaT, kB=kB, grid=grid)

    # Logging
    hills_file = "hills.dat"
    callback = MetaDLogger(hills_file, stride) if args.log else None

    # Logging trajectory
    log = {
        "kT": np.zeros((timesteps,)),
        "H": np.zeros((timesteps,)),
        "position": np.zeros((timesteps // write_every,) + R.shape),
    }

    tic = time.perf_counter()
    context_args = {
        "total_energy_fn": total_energy_fn,
        "shift_fn": shift_fn,
        "dt": dt,
        "kT": kT,
        "state": Rfinal,
        "box": box,
        "log": log,
    }
    run_result = pysages.run(
        method, generate_simulation, timesteps, callback, context_args, None, key=key, R=Rfinal
    )
    toc = time.perf_counter()
    print(f"Completed the simulation in {toc - tic:0.4f} seconds.")

    #### # Analysis: Calculate free energy using the deposited bias potential

    #### # generate CV values on a grid to evaluate bias potential
    #### plot_grid = pysages.Grid(lower=(0), upper=(2), shape=(50), periodic=True)
    #### xi = (compute_mesh(plot_grid) + 1) / 2 * plot_grid.size + plot_grid.lower

    #### # determine bias factor depending on method (for standard = 1 and for well-tempered = (T+deltaT)/deltaT)
    #### alpha = (
    ####     1
    ####     if method.deltaT is None
    ####     else (T.value_in_unit(unit.kelvin) + method.deltaT) / method.deltaT
    #### )
    #### kT = kB * T.value_in_unit(unit.kelvin)

    #### # extract metapotential function from result
    #### result = pysages.analyze(run_result)
    #### metapotential = result["metapotential"]

    #### # report in kT and set min free energy to zero
    #### A = metapotential(xi) * -alpha / kT
    #### A = A - A.min()
    #### A = A.reshape(plot_grid.shape)

    #### # plot and save free energy to a PNG file
    #### fig, ax = plt.subplots(dpi=120)

    #### ax.plot(xi, A, colors="k")
    #### ax.set_xlabel(r"$\Distance$")
    #### ax.set_ylabel(r"$A~[k_{B}T]$")

    #### fig.savefig("adp-2particles-ljfluid.png", dpi=fig.dpi)

    #### return result


# %%
if __name__ == "__main__":
    main(sys.argv[1:])
