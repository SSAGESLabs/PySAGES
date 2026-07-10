#!/usr/bin/env python3
"""
Well-tempered metadynamics of ethane dihedral angle: PySAGES + GPUMD

Author: Jaafar Mehrez
(Shanghai Jiao Tong University, Shanghai, China;
 HPQC Labs, Waterloo, Canada;
 jaafarmehrez@sjtu.edu.cn, jaafar@hpqc.org)

This script uses *well-tempered* metadynamics to compute the free energy
surface (FES) along the H-C-C-H dihedral angle of ethane.

For background on ethane conformations, see:
https://chem.libretexts.org/Courses/Athabasca_University/...

Before running:
1. Build gpumd.so:   cd GPUMD/src && make pygpumd
2. Ensure gpumd.so is on PYTHONPATH
3. Have a GPUMD simulation directory with run.in and model.xyz

Usage:
    python ethane-metad.py

SPDX-License-Identifier: MIT
"""

import os
import sys
import time

import numpy as np

"""Ensure the compiled GPUMD module is importable"""
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GPUMD_SRC = os.path.join(_SCRIPT_DIR, "GPUMD", "src")
if os.path.isdir(_GPUMD_SRC) and _GPUMD_SRC not in sys.path:
    sys.path.insert(0, _GPUMD_SRC)

import gpumd
import jax

jax.config.update("jax_enable_x64", True)

import pysages
from pysages.approxfun import compute_mesh
from pysages.backends.core import SamplingContext
from pysages.colvars import DihedralAngle
from pysages.methods import MetaDLogger, Metadynamics
from pysages.methods.core import Result

"""Simulation setup"""
SIMULATION_DIR = "/path/to/your/gpumd/simulation"
RUN_IN_PATH = os.path.join(SIMULATION_DIR, "run.in")

if not os.path.isfile(RUN_IN_PATH):
    raise FileNotFoundError(
        f"Cannot find {RUN_IN_PATH}. Please create a GPUMD simulation directory first."
    )


def generate_simulation(**kwargs):
    """Return a GPUMD simulation object (backend context)."""
    os.chdir(SIMULATION_DIR)
    return gpumd.Simulation(RUN_IN_PATH)


"""
Collective variable: H-C-C-H dihedral angle

 From model.xyz (ethane, 8 atoms):
   0  C   (carbon 1)
   1  H   (hydrogen on C1)
   2  H   (hydrogen on C1)
   3  H   (hydrogen on C1)
   4  C   (carbon 2)
   5  H   (hydrogen on C2)
   6  H   (hydrogen on C2)
   7  H   (hydrogen on C2)

Dihedral angle: H(1) -- C(0) -- C(4) -- H(5)
"""

pi = np.pi
cvs = [DihedralAngle([1, 0, 4, 5])]

"""Well-tempered metadynamics parameters"""

height = 0.02  # Initial Gaussian height in eV (GPUMD energy unit)
sigma = [0.3]  # Gaussian width in radians
stride = 100  # Deposit a hill every 100 steps
timesteps = 1000_000  # Total simulation steps
ngauss = timesteps // stride + 1
deltaT = 1500.0  # Fictitious temperature in Kelvin (5x 300 K)
kB = 8.617333262e-5  # Boltzmann constant in eV/K

grid = pysages.Grid(
    lower=(-pi,),
    upper=(pi,),
    shape=(200,),
    periodic=True,
)
method = Metadynamics(
    cvs,
    height,
    sigma,
    stride,
    ngauss,
    grid=grid,
    deltaT=deltaT,
    kB=kB,
)
hills_file = "hills_ethane_wt.dat"
callback = MetaDLogger(hills_file, stride)

print("Starting ethane dihedral well-tempered metadynamics...")
print(f"  CV: dihedral angle H(1)-C(0)-C(4)-H(5)")
print(f"  Grid: [{-pi:.3f}, {pi:.3f}] rad  (periodic)")
print(f"  Hills: height={height} eV, sigma={sigma[0]} rad, stride={stride}")
print(f"  Well-tempered: deltaT={deltaT} K, kB={kB:.4e} eV/K")
print(f"  Total steps: {timesteps} -> ~{ngauss} hills")

sim = generate_simulation()
tic = time.perf_counter()

sampling_context = SamplingContext(method, lambda: sim)
with sampling_context:
    sampling_context.run(timesteps)
    sampler = sampling_context.sampler
toc = time.perf_counter()
print(f"Completed in {toc - tic:0.1f} seconds.")

if hasattr(sampler, "print_timings"):
    sampler.print_timings()

run_result = Result(
    method,
    [sampler.state],
    None if sampler.callback is None else [sampler.callback],
    [sampler.take_snapshot()],
)

T = 300.0
plot_grid = pysages.Grid(
    lower=(-pi,),
    upper=(pi,),
    shape=(400,),
    periodic=True,
)
xi = compute_mesh(plot_grid)
result = pysages.analyze(run_result)
metapotential = result["metapotential"]
alpha = 1.0 + T / deltaT
A = -alpha * metapotential(xi)
A = A - A.min()

output = np.column_stack((xi.flatten() * 180 / pi, A.flatten()))
np.savetxt(
    "fes_ethane_wt.dat",
    output,
    header="dihedral_angle_deg  free_energy_eV",
    comments="",
    fmt="%.6f",
)
print("Free energy surface saved to fes_ethane_wt.dat")

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xi.flatten() * 180 / pi, A.flatten(), lw=2, color="steelblue")
    ax.axhline(y=0, color="gray", ls="--", lw=0.5)
    ax.annotate(
        "staggered (min)",
        xy=(60, 0),
        xytext=(60, 0.05),
        ha="center",
        fontsize=9,
        color="green",
    )
    ax.annotate(
        "eclipsed (max)",
        xy=(0, A.max()),
        xytext=(0, A.max() + 0.02),
        ha="center",
        fontsize=9,
        color="red",
    )

    ax.set_xlabel(r"Dihedral angle $\phi$ (degrees)")
    ax.set_ylabel(r"Free energy $\Delta G$ (eV)")
    ax.set_title("Ethane rotational free energy (well-tempered metadynamics)")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])

    fig.tight_layout()
    fig.savefig("fes_ethane_wt.png", dpi=150)
    print("Plot saved to fes_ethane_wt.png")
except ImportError:
    print("matplotlib not available; skipping plot.")

"""Sanity check: barrier height"""
barrier = A.max() - A.min()
print(f"\nFES barrier height: {barrier:.4f} eV")
print(f"  (Literature value for ethane: ~0.12 eV = ~12 kJ/mol)")
print(f"  Well-tempered scaling factor alpha = {alpha:.3f}")

if barrier < 0.05:
    print("WARNING: barrier seems very low. Try increasing timesteps or")
    print("         decreasing hill height/sigma for better resolution.")
elif barrier > 0.5:
    print("WARNING: barrier seems very high. Check units or hill parameters.")
else:
    print("Barrier height is in a physically reasonable range.")

print("\nDone.")
