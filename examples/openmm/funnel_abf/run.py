#!/usr/bin/env python3

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
from parmed import load_file
from parmed.openmm import StateDataReporter

import pysages
from pysages.backends import SamplingContext
from pysages.colvars import Projection_on_Axis_mobile

# %%
from pysages.methods import CVRestraints, Funnel_ABF, Funnel_Logger, get_funnel_force
from pysages.utils import try_import

# %%
openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")

# %%
pi = np.pi
kB = 0.008314462618  # kJ/mol


# %%
def generate_simulation(T=298.15 * unit.kelvin, dt=1.0 * unit.femtoseconds):
    print("Loading AMBER files...")
    ala2_solv = load_file("complex-wat.prmtop", "complex-wat-prod.rst")
    system = ala2_solv.createSystem(
        nonbondedMethod=app.PME,
        rigidWater=True,
        switchDistance=1.0 * unit.nanometer,
        nonbondedCutoff=1.2 * unit.nanometer,
        constraints=app.HBonds,
    )
    # Create the integrator to do Langevin dynamics
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,  # Temperature of heat bath
        1.0 / unit.picoseconds,  # Friction coefficient
        1.0 * unit.femtoseconds,  # Time step
    )

    # Define the platform to use; CUDA, OpenCL, CPU, or Reference. Or do not specify
    platform = openmm.Platform.getPlatformByName("CPU")
    # Create the Simulation object

    sim = app.Simulation(ala2_solv.topology, system, integrator, platform)

    # Set the particle positions
    sim.context.setPositions(ala2_solv.positions)
    sim.reporters.append(app.PDBReporter("output.pdb", 200000))
    sim.reporters.append(app.DCDReporter("output.dcd", 200000))
    sim.reporters.append(StateDataReporter("data.txt", 20000, step=True, separator=" "))

    return sim


# %%
# functions for ploting and storing data
def plot_energy(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Free energy $[\\epsilon]$")

    free_energy = np.asarray(result["free_energy"])
    x = np.asarray(result["mesh"])
    ax.plot(x, free_energy, color="teal")

    fig.savefig("energy.png")


def plot_forces(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Forces $[\\epsilon]$")

    forces = np.asarray(result["mean_force"])
    x = np.asarray(result["mesh"])
    ax.plot(x, forces, color="teal")

    fig.savefig("forces.png")


def plot_histogram(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Histogram $[\\epsilon]$")

    hist = np.asarray(result["histogram"])
    x = np.asarray(result["mesh"])
    ax.plot(x, hist, color="teal")

    fig.savefig("histogram2.png")


def save_energy_forces(result):
    Energy = np.asarray(result["free_energy"])
    Forces = np.asarray(result["mean_force"])
    Grid = np.asarray(result["mesh"])
    hist = np.asarray(result["histogram"])
    np.savetxt("FES.csv", np.column_stack([Grid, Energy]))
    np.savetxt("Forces.csv", np.column_stack([Grid, Forces]))
    np.savetxt("Histogram.csv", np.column_stack([Grid, hist]))


#    numpy.savetxt("Forces.csv", numpy.column_stack([Grid, Forces]))

# %%
# %%
# %%
host = list(range(0, 144))
ligand = list(range(144, 168))
weights_host = np.ones(len(host)) / len(host)
weights_lig = np.ones(len(ligand)) / len(ligand)
anchor = 89
indices_sys = [ligand, host, [anchor]]
A = [4.6879, 4.8335, 1.12045]
B = [4.7829, 5.7927, 2.8729]
# anchor = 89
box = [5.5822, 5.5095, 5.4335]
Z_0 = 0.610225
Zcc = 1.2
R_cyl = 0.6
k_cone = 10000.0
k_cv = 0.0
cv_min = 0.0
cv_max = 2.0
cv_buffer = 0.05
coordinates = open("referencesh.pdb", "r")
ref_loop1 = []
for line in coordinates:
    lista = line.split()
    id = lista[0]
    if id == "ATOM":
        atomid = int(lista[1])
        residue = int(lista[4])
        position = lista[5:8]
        temp_pos = []
        for p in position:
            temp_pos.append(float(p) / 10.0)
        ref_loop1.append(temp_pos)
coordinates.close()
# print(ref_loop1)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# print(indices_sys)
cvs = (
    Projection_on_Axis_mobile(
        indices_sys,
        references=ref_loop1,
        weights_lig=None,
        weights_prot=None,
        A=A,
        B=B,
        box=box,
    ),
)

grid = pysages.Grid(lower=(cv_min,), upper=(cv_max,), shape=(32,), periodic=False)

restraints = CVRestraints(lower=(-0.2,), upper=(2.0,), kl=(10000.0,), ku=(10000.0,))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
funnel_force = get_funnel_force(
    indices_sys, ref_loop1, A, B, Zcc, Z_0, R_cyl, k_cone, k_cv, cv_min, cv_max, cv_buffer, box
)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
height = 0.2
sigma = 0.02
stride = 500
timesteps = 500
ngauss = timesteps // stride + 1
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

methodo = Funnel_ABF(cvs, grid=grid, N=1000, ext_force=funnel_force, restraints=restraints)
funnel_file = "funnel.dat"
callback = Funnel_Logger(funnel_file, 10)
sampling_context = SamplingContext(methodo, generate_simulation, callback)
state = pysages.run(sampling_context, timesteps)
# for restart simulation
# with open("restart1.pickle", "rb") as f:
#    state = pickle.load(f)
# state = pysages.run(state, generate_simulation, timesteps)
# save pickle file
with open("restart1.pickle", "wb") as f:
    pickle.dump(state, f)
topology = (8, 8)
result = pysages.analyze(state, topology=topology)
plot_energy(result)
plot_forces(result)
plot_histogram(result)
save_energy_forces(result)

# %%
# %%
