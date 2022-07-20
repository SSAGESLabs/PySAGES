#!/usr/bin/env python3


# %%
from pysages.colvars import DihedralAngle
from pysages.methods import ABF
from pysages.utils import try_import
import matplotlib.pyplot as plt
import numpy
import pysages

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


# %%
pi = numpy.pi

adp_pdb = "../../inputs/alanine-dipeptide/adp-explicit.pdb"
T = 298.15 * unit.kelvin
dt = 2.0 * unit.femtoseconds

# %%
def generate_simulation(pdb_filename=adp_pdb, T=T, dt=dt):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml", "tip3p.xml")
    cutoff_distance = 1.0 * unit.nanometer
    topology = pdb.topology

    system = ff.createSystem(
        topology, constraints=app.HBonds, nonbondedMethod=app.PME, nonbondedCutoff=cutoff_distance
    )

    # Set dispersion correction use.
    forces = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        forces[force.__class__.__name__] = force

    forces["NonbondedForce"].setUseDispersionCorrection(True)
    forces["NonbondedForce"].setEwaldErrorTolerance(1.0e-5)

    positions = pdb.getPositions(asNumpy=True)

    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)

    # platform = openmm.Platform.getPlatformByName(platform)
    # simulation = app.Simulation(topology, system, integrator, platform)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    return simulation


# %%
# helping function for plotting results
def plot_energy(result):
    surface = numpy.asarray(result["free_energy"])
    fig, ax = plt.subplots()
    im = ax.imshow(
        surface, interpolation="bicubic", origin="lower", extent=[-pi, pi, -pi, pi], aspect=1
    )
    ax.contour(surface, levels=15, linewidths=0.75, colors="k", extent=[-pi, pi, -pi, pi])
    plt.colorbar(im)
    fig.savefig("energy.pdf")


# %%
# %%
def plot_histogram(result):
    surface = numpy.asarray(result["histogram"]) / numpy.nanmax(numpy.asarray(result["histogram"]))
    fig, ax = plt.subplots()
    im = ax.imshow(
        surface, interpolation="bicubic", origin="lower", extent=[-pi, pi, -pi, pi], aspect=1
    )
    ax.contour(surface, levels=15, linewidths=0.75, colors="k", extent=[-pi, pi, -pi, pi])
    plt.colorbar(im)
    fig.savefig("histogram.pdf")


# %%
# %%
def plot_forces(result):
    fig, ax = plt.subplots()

    ax.set_xlabel("CV")
    ax.set_ylabel("Forces $[\\epsilon]$")

    forces = numpy.asarray(result["mean_force"])
    x = numpy.asarray(result["mesh"])
    plt.quiver(x, forces, width=(0.0002 * (x[x.shape[0] - 1, 0] - x[0, 0])), headwidth=3)

    fig.savefig("forces.pdf")


# Stores forces and free energies for post-analysis
def save_energy_forces(result):
    energy = numpy.asarray(result["free_energy"])
    forces = numpy.asarray(result["mean_force"])
    grid = numpy.asarray(result["mesh"])
    numpy.savetxt("FES.csv", numpy.hstack([grid, energy.reshape(-1, 1)]))
    numpy.savetxt("Forces.csv", numpy.hstack([grid, forces.reshape(-1, grid.shape[1])]))


# %%
def main():
    cvs = [DihedralAngle((4, 6, 8, 14)), DihedralAngle((6, 8, 14, 16))]
    grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(32, 32), periodic=True)
    method = ABF(cvs, grid)

    raw_result = pysages.run(method, generate_simulation, 25)
    result = pysages.analyze(raw_result, topology=(14,))

    plot_energy(result)
    plot_histogram(result)
    save_energy_forces(result)


# %%
if __name__ == "__main__":
    main()
