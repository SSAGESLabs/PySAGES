#!/usr/bin/env python3

"""
Metadynamics simulation of Alanine Dipeptide in water with OpenMM and PySAGES using
Permutation Invariant Vector (PIV) as CVs.

Example command to run the simulation `python3 alanine-dipeptide.py --time-steps 1000`
For other supported commandline parameters, check `python3 alanine-dipeptide.py --help`
"""


# %%
import argparse
import os
import sys
import time

import numpy
import pysages

from pysages.colvars import PIV
from pysages.methods import Metadynamics, MetaDLogger
from pysages.utils import try_import
from pysages.approxfun import compute_mesh

import numpy as onp
from jax import numpy as np
from jax_md.partition import neighbor_list as nlist, space
from jax_md.partition import NeighborListFormat

import matplotlib.pyplot as plt

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


# %%
pi = numpy.pi
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
kB = kB.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)

T = 298.15 * unit.kelvin
dt = 2.0 * unit.femtoseconds
adp_pdb = os.path.join(os.pardir, os.pardir, os.pardir, "inputs", "alanine-dipeptide", "adp-explicit.pdb")


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

    integrator.setRandomNumberSeed(42)

    # platform = openmm.Platform.getPlatformByName(platform)
    # simulation = app.Simulation(topology, system, integrator, platform)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    simulation.reporters.append(app.PDBReporter("output.pdb", 1000))
    simulation.reporters.append(
        app.StateDataReporter("log.dat", 1000, step=True, potentialEnergy=True, temperature=True)
    )

    return simulation

def gen_neighbor_list(pdb_filename, custom_mask_function):

    pdb = app.PDBFile(pdb_filename)
    top = pdb.getTopology()
    positions = np.array(pdb.getPositions(asNumpy=True), dtype=np.float32)

    dr_threshold = 0.5
    box_size = 3
    nl_cutoff = 1
    displacement_fn, shift_fn = space.periodic(box_size)
    neighbor_list_fn = nlist(displacement_fn, box_size, nl_cutoff, dr_threshold, capacity_multiplier=0.5,
                            custom_mask_function=custom_mask_function, format=NeighborListFormat.Dense)
    neighbors = neighbor_list_fn.allocate(positions)

    return neighbors


def gen_atomtype_lists(pdb_filename=adp_pdb, atomtypes=['C', 'N', 'O'], solventname='HOH'):
    
    pdb = app.PDBFile(pdb_filename)
    top = pdb.getTopology()
    
    # separate each atom type of interest - solute and solvent oxygen into a list
    solute_list = []
    for residue in top.residues():
        if residue.name != solventname:
            for atomtype in atomtypes:
                for atom in residue.atoms():
                    if atom.name.startswith(atomtype):
                        solute_list.append([int(atom.id)-1])
            
          
    solute_atoms = []
    oxygen_list = []
    hydrogen_dict = {}
    hydrogen_array = np.ones((pdb.topology.getNumAtoms(), 2))*(-1000)
    for residue in top.residues():
        if residue.name == solventname:
            for atom in residue.atoms():
                if atom.name.startswith('O'):
                    oxygen_list.append(int(atom.id)-1)
                    hatom_list = []
                    for bond in residue.bonds():
                        if bond.atom1.id == atom.id:
                            hatom_list.append(int(bond.atom2.id)-1)
                        elif bond.atom2.id == atom.id:
                            hatom_list.append(int(bond.atom1.id)-1)
                    hydrogen_dict[int(atom.id)-1] = hatom_list
                    hydrogen_array = hydrogen_array.at[int(atom.id)-1].set(np.array(hatom_list))
                if atom.name.startswith('H'):
                    solute_atoms.append(int(atom.id)-1)
        else:
            for atom in residue.atoms():
                solute_atoms.append(int(atom.id)-1)
                
                        
         
            #atom_indices.append(oxygen_list)
    
    print("oxygen list")
    print(oxygen_list)
    
    print("hydrogen dict")
    print(hydrogen_dict[22])
    
    print("hydrogen array")
    print(hydrogen_array)
    
    print("\n")
    
    num_atoms = top.getNumAtoms()
    natom_types = len(atomtypes) + 1
    
    return solute_atoms, solute_list, oxygen_list, hydrogen_array, num_atoms, natom_types
    

def gen_atompair_list(atom_lists, natom_types, exclude_atomtype_pairindices):

    position_pairs = []
    for i in range(natom_types):
        
        for j in range(i, natom_types):
            
            for i_particle in range(len(atom_lists[i])):
                
                for j_particle in range(len(atom_lists[j])):
                    
                    if i == j and j_particle <= i_particle:
                        continue
                        
                    if [i, j] in exclude_atomtype_pairindices:
                        continue
                            
                    position_pairs.append([i, atom_lists[i][i_particle], j, atom_lists[j][j_particle]])
                    
    return np.array(position_pairs)


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
    
    atom_indices, solute_list, oxygen_list, hydrogen_array, num_atoms, natom_types = gen_atomtype_lists()
    exclude_atomtype_pairindices = [ [1, 1], [1, 2] ]
    
    position_pairs = gen_atompair_list(solute_list, natom_types, exclude_atomtype_pairindices)
    
    all_atoms = list(onp.arange(num_atoms))
    #atom_indices.append(all_atoms)
    
    print("solute atoms ")
    print(np.array(solute_list).flatten())
    
    print(atom_indices)
    
    def filter_solvent_neighbors(idx):
        mask = np.isin(idx, np.array(atom_indices), invert=True)
        return np.where(mask, idx, num_atoms)
        #return idx

    cvs = [PIV( all_atoms,
                position_pairs,
                solute_list,
                oxygen_list,
                hydrogen_array,
                {'r_0': 0.4, 'd_0': 2.3, 'n': 3, 'm': 6},
                {'neighbor_list': gen_neighbor_list(adp_pdb,
                                                    filter_solvent_neighbors)})]

    height = 1.2  # kJ/mol
    sigma = [0.35, 0.35]  # radians
    deltaT = 5000 if args.well_tempered else None
    stride = 500  # frequency for depositing gaussians
    timesteps = args.time_steps
    ngauss = timesteps // stride  # total number of gaussians

    ## Grid for storing bias potential and its gradient
    #grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(50, 50), periodic=True)
    #grid = grid if args.use_grids else None

    # Method
    method = Metadynamics(cvs, height, sigma, stride, ngauss, deltaT=deltaT, kB=kB) # grid=grid)

    # Logging
    hills_file = "hills.dat"
    callback = MetaDLogger(hills_file, stride) if args.log else None

    tic = time.perf_counter()
    run_result = pysages.run(method, generate_simulation, timesteps, callback)
    toc = time.perf_counter()
    print(f"Completed the simulation in {toc - tic:0.4f} seconds.")

    # Analysis: Calculate free energy using the deposited bias potential

    # generate CV values on a grid to evaluate bias potential
    #plot_grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(64, 64), periodic=True)
    #xi = (compute_mesh(plot_grid) + 1) / 2 * plot_grid.size + plot_grid.lower

    # determine bias factor depending on method (for standard = 1 and for well-tempered = (T+deltaT)/deltaT)
    #alpha = (
    #    1
    #    if method.deltaT is None
    #    else (T.value_in_unit(unit.kelvin) + method.deltaT) / method.deltaT
    #)
    #kT = kB * T.value_in_unit(unit.kelvin)

    ## extract metapotential function from result
    #result = pysages.analyze(run_result)
    #metapotential = result["metapotential"]

    ## report in kT and set min free energy to zero
    #A = metapotential(xi) * -alpha / kT
    #A = A - A.min()
    #A = A.reshape(plot_grid.shape)

    ## plot and save free energy to a PNG file
    #fig, ax = plt.subplots(dpi=120)

    #im = ax.imshow(A, interpolation="bicubic", origin="lower", extent=[-pi, pi, -pi, pi])
    #ax.contour(A, levels=12, linewidths=0.75, colors="k", extent=[-pi, pi, -pi, pi])
    #ax.set_xlabel(r"$\phi$")
    #ax.set_ylabel(r"$\psi$")

    #cbar = plt.colorbar(im)
    #cbar.ax.set_ylabel(r"$A~[k_{B}T]$", rotation=270, labelpad=20)

    #fig.savefig("adp-fe.png", dpi=fig.dpi)

    #return result


# %%
if __name__ == "__main__":

    main(sys.argv[1:])
