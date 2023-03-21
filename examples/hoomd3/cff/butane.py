#!/usr/bin/env python3

"""
CFF simulation of a butane molecule in HOOMD-blue and PySAGES.
"""


# %%
import argparse
import sys
import time
from math import pi, sqrt

import gsd
import gsd.hoomd
import hoomd
import hoomd.md
import numpy as np

import pysages
from pysages import Grid
from pysages.colvars import DihedralAngle
from pysages.methods import CFF

# %%
kT = 0.596161
dt = 0.02045


# %%
def generate_context(kT=kT, dt=dt, **kwargs):
    sim = hoomd.Simulation(
        device=kwargs.get("context", hoomd.device.CPU()), seed=kwargs.get("seed", 1)
    )
    snapshot = gsd.hoomd.Frame()
    snapshot.particles.N = N = 14
    snapshot.configuration.box = [41, 41, 41, 0, 0, 0]
    snapshot.particles.types = ["C", "H"]
    snapshot.bonds.types = ["CC", "CH"]
    snapshot.angles.types = ["CCC", "CCH", "HCH"]
    snapshot.dihedrals.types = ["CCCC", "HCCC", "HCCH"]
    snapshot.pairs.types = ["CCCC", "HCCC", "HCCH"]
    snapshot.particles.typeid = np.zeros(N, dtype=int)
    snapshot.particles.position = np.zeros((N, 3))
    snapshot.particles.mass = np.zeros(N, dtype=float)
    snapshot.particles.charge = np.zeros(N, dtype=float)
    snapshot.particles.typeid[0] = 0
    snapshot.particles.typeid[1:4] = 1
    snapshot.particles.typeid[4] = 0
    snapshot.particles.typeid[5:7] = 1
    snapshot.particles.typeid[7] = 0
    snapshot.particles.typeid[8:10] = 1
    snapshot.particles.typeid[10] = 0
    snapshot.particles.typeid[11:14] = 1

    positions = np.array(
        [
            [-2.990196, 0.097881, 0.000091],
            [-2.634894, -0.911406, 0.001002],
            [-2.632173, 0.601251, -0.873601],
            [-4.060195, 0.099327, -0.000736],
            [-2.476854, 0.823942, 1.257436],
            [-2.832157, 1.833228, 1.256526],
            [-2.834877, 0.320572, 2.131128],
            [-0.936856, 0.821861, 1.258628],
            [-0.578833, 1.325231, 0.384935],
            [-0.581553, -0.187426, 1.259538],
            [-0.423514, 1.547922, 2.515972],
            [-0.781537, 1.044552, 3.389664],
            [0.646485, 1.546476, 2.516800],
            [-0.778816, 2.557208, 2.515062],
        ]
    )

    reference_box_low_coords = np.array([-22.206855, -19.677099, -19.241968])
    box_low_coords = np.array([-41.0 / 2, -41.0 / 2, -41.0 / 2])
    positions += box_low_coords - reference_box_low_coords

    snapshot.particles.position[:] = positions[:]

    mC = 12.00
    mH = 1.008

    snapshot.particles.mass[:] = [
        mC,
        mH,
        mH,
        mH,  # grouped by carbon atoms
        mC,
        mH,
        mH,
        mC,
        mH,
        mH,
        mC,
        mH,
        mH,
        mH,
    ]

    reference_charges = np.array(
        [
            -0.180000,
            0.060000,
            0.060000,
            0.060000,  # grouped by carbon atoms
            -0.120000,
            0.060000,
            0.060000,
            -0.120000,
            0.060000,
            0.060000,
            -0.180000,
            0.060000,
            0.060000,
            0.060000,
        ]
    )

    charge_conversion = 18.22262
    snapshot.particles.charge[:] = charge_conversion * reference_charges[:]
    snapshot.bonds.N = 13
    snapshot.bonds.typeid = np.zeros(13, dtype=int)
    snapshot.bonds.typeid[0:3] = 1
    snapshot.bonds.typeid[3] = 0
    snapshot.bonds.typeid[4:6] = 1
    snapshot.bonds.typeid[6] = 0
    snapshot.bonds.typeid[7:9] = 1
    snapshot.bonds.typeid[9] = 0
    snapshot.bonds.typeid[10:13] = 1

    snapshot.bonds.group = np.zeros((13, 2), dtype=int)
    snapshot.bonds.group[:] = [
        [0, 2],
        [0, 1],
        [0, 3],
        [0, 4],  # grouped by carbon atoms
        [4, 5],
        [4, 6],
        [4, 7],
        [7, 8],
        [7, 9],
        [7, 10],
        [10, 11],
        [10, 12],
        [10, 13],
    ]
    snapshot.angles.N = 24
    snapshot.angles.typeid = np.zeros(24, dtype=int)
    snapshot.angles.typeid[0:2] = 2
    snapshot.angles.typeid[2] = 1
    snapshot.angles.typeid[3] = 2
    snapshot.angles.typeid[4:8] = 1
    snapshot.angles.typeid[8] = 0
    snapshot.angles.typeid[9] = 2
    snapshot.angles.typeid[10:14] = 1
    snapshot.angles.typeid[14] = 0
    snapshot.angles.typeid[15] = 2
    snapshot.angles.typeid[16:21] = 1
    snapshot.angles.typeid[21:24] = 2

    snapshot.angles.group = np.zeros((24, 3), dtype=int)
    snapshot.angles.group[:] = [
        [1, 0, 2],
        [2, 0, 3],
        [2, 0, 4],  # grouped by carbon atoms
        [1, 0, 3],
        [1, 0, 4],
        [3, 0, 4],
        # ---
        [0, 4, 5],
        [0, 4, 6],
        [0, 4, 7],
        [5, 4, 6],
        [5, 4, 7],
        [6, 4, 7],
        # ---
        [4, 7, 8],
        [4, 7, 9],
        [4, 7, 10],
        [8, 7, 9],
        [8, 7, 10],
        [9, 7, 10],
        # ---
        [7, 10, 11],
        [7, 10, 12],
        [7, 10, 13],
        [11, 10, 12],
        [11, 10, 13],
        [12, 10, 13],
    ]
    snapshot.dihedrals.N = 27
    snapshot.dihedrals.typeid = np.zeros(27, dtype=int)
    snapshot.dihedrals.typeid[0:2] = 2
    snapshot.dihedrals.typeid[2] = 1
    snapshot.dihedrals.typeid[3:5] = 2
    snapshot.dihedrals.typeid[5] = 1
    snapshot.dihedrals.typeid[6:8] = 2
    snapshot.dihedrals.typeid[8:11] = 1
    snapshot.dihedrals.typeid[11] = 0
    snapshot.dihedrals.typeid[12:14] = 2
    snapshot.dihedrals.typeid[14] = 1
    snapshot.dihedrals.typeid[15:17] = 2
    snapshot.dihedrals.typeid[17:21] = 1
    snapshot.dihedrals.typeid[21:27] = 2

    snapshot.dihedrals.group = np.zeros((27, 4), dtype=int)
    snapshot.dihedrals.group[:] = [
        [2, 0, 4, 5],
        [2, 0, 4, 6],
        [2, 0, 4, 7],  # grouped by pairs of central atoms
        [1, 0, 4, 5],
        [1, 0, 4, 6],
        [1, 0, 4, 7],
        [3, 0, 4, 5],
        [3, 0, 4, 6],
        [3, 0, 4, 7],
        # ---
        [0, 4, 7, 8],
        [0, 4, 7, 9],
        [0, 4, 7, 10],
        [5, 4, 7, 8],
        [5, 4, 7, 9],
        [5, 4, 7, 10],
        [6, 4, 7, 8],
        [6, 4, 7, 9],
        [6, 4, 7, 10],
        # ---
        [4, 7, 10, 11],
        [4, 7, 10, 12],
        [4, 7, 10, 13],
        [8, 7, 10, 11],
        [8, 7, 10, 12],
        [8, 7, 10, 13],
        [9, 7, 10, 11],
        [9, 7, 10, 12],
        [9, 7, 10, 13],
    ]
    snapshot.pairs.N = 27
    snapshot.pairs.typeid = np.zeros(27, dtype=int)
    snapshot.pairs.typeid[0:1] = 0
    snapshot.pairs.typeid[1:11] = 1
    snapshot.pairs.typeid[11:27] = 2
    snapshot.pairs.group = np.zeros((27, 2), dtype=int)
    snapshot.pairs.group[:] = [
        # CCCC
        [0, 10],
        # HCCC
        [0, 8],
        [0, 9],
        [5, 10],
        [6, 10],
        [1, 7],
        [2, 7],
        [3, 7],
        [11, 4],
        [12, 4],
        [13, 4],
        # HCCH
        [1, 5],
        [1, 6],
        [2, 5],
        [2, 6],
        [3, 5],
        [3, 6],
        [5, 8],
        [6, 8],
        [5, 9],
        [6, 9],
        [8, 11],
        [8, 12],
        [8, 13],
        [9, 11],
        [9, 12],
        [9, 13],
    ]
    snapshot.particles.validate()
    sim.create_state_from_snapshot(snapshot, domain_decomposition=(None, None, None))
    exclusions = ["bond", "1-3", "1-4"]
    nl = hoomd.md.nlist.Cell(buffer=0.4, exclusions=exclusions)
    lj = hoomd.md.pair.LJ(nlist=nl, default_r_cut=12.0)
    lj.params[("C", "C")] = {"epsilon": 0.07, "sigma": 3.55}
    lj.params[("H", "H")] = {"epsilon": 0.03, "sigma": 2.42}
    lj.params[("C", "H")] = {"epsilon": sqrt(0.07 * 0.03), "sigma": sqrt(3.55 * 2.42)}

    coulomb = hoomd.md.long_range.pppm.make_pppm_coulomb_forces(
        nlist=nl, resolution=[64, 64, 64], order=6, r_cut=12.0
    )

    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["CC"] = dict(k=2 * 268.0, r0=1.529)
    harmonic.params["CH"] = dict(k=2 * 340.0, r0=1.09)

    angle = hoomd.md.angle.Harmonic()
    angle.params["CCC"] = dict(k=2 * 58.35, t0=112.7 * pi / 180)
    angle.params["CCH"] = dict(k=2 * 37.5, t0=110.7 * pi / 180)
    angle.params["HCH"] = dict(k=2 * 33.0, t0=107.8 * pi / 180)

    dihedral = hoomd.md.dihedral.OPLS()
    dihedral.params["CCCC"] = dict(k1=1.3, k2=-0.05, k3=0.2, k4=0.0)
    dihedral.params["HCCC"] = dict(k1=0.0, k2=0.0, k3=0.3, k4=0.0)
    dihedral.params["HCCH"] = dict(k1=0.0, k2=0.0, k3=0.3, k4=0.0)

    lj_special_pairs = hoomd.md.special_pair.LJ()
    lj_special_pairs.params["CCCC"] = dict(epsilon=0.07, sigma=3.55)
    lj_special_pairs.params["HCCH"] = dict(epsilon=0.03, sigma=2.42)
    lj_special_pairs.params["HCCC"] = dict(epsilon=sqrt(0.07 * 0.03), sigma=sqrt(3.55 * 2.42))
    lj_special_pairs.r_cut["CCCC"] = 12.0
    lj_special_pairs.r_cut["HCCC"] = 12.0
    lj_special_pairs.r_cut["HCCH"] = 12.0
    coulomb_special_pairs = hoomd.md.special_pair.Coulomb()
    coulomb_special_pairs.params["CCCC"] = dict(alpha=0.5)
    coulomb_special_pairs.params["HCCC"] = dict(alpha=0.5)
    coulomb_special_pairs.params["HCCH"] = dict(alpha=0.5)
    coulomb_special_pairs.r_cut["HCCH"] = 12.0
    coulomb_special_pairs.r_cut["CCCC"] = 12.0
    coulomb_special_pairs.r_cut["HCCC"] = 12.0
    integrator = hoomd.md.Integrator(dt=dt)
    integrator.forces.append(lj)
    integrator.forces.append(coulomb[0])
    integrator.forces.append(coulomb[1])
    integrator.forces.append(harmonic)
    integrator.forces.append(angle)
    integrator.forces.append(dihedral)
    integrator.forces.append(lj_special_pairs)
    integrator.forces.append(coulomb_special_pairs)
    nvt = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    return sim


# %%
# %%
def get_args(argv):
    available_args = [
        ("time-steps", "t", int, 5e5, "Number of simulation steps"),
        ("train-freq", "f", int, 5e3, "Frequency for neural network training"),
    ]
    parser = argparse.ArgumentParser(description="Example script to run CFF")
    for name, short, T, val, doc in available_args:
        parser.add_argument("--" + name, "-" + short, type=T, default=T(val), help=doc)

    return parser.parse_args(argv)


# %%
def main(argv=[]):
    args = get_args(argv)

    cvs = [DihedralAngle([0, 4, 7, 10])]
    grid = Grid(lower=(-pi,), upper=(0,), shape=(64,), periodic=True)
    topology = (8,)

    method = CFF(cvs, grid, topology, kT, train_freq=args.train_freq)

    timesteps = args.time_steps
    tic = time.perf_counter()
    run_result = pysages.run(method, generate_context, timesteps)
    toc = time.perf_counter()
    print(f"Completed the simulation in {toc - tic:0.4f} seconds.")

    result = pysages.analyze(run_result)
    return result["free_energy"]


# %%
if __name__ == "__main__":
    main(sys.argv[1:])
