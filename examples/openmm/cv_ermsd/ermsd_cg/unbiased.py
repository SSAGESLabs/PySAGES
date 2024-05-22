#!/usr/bin/env python

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

import pysages
from pysages.colvars.orientation import ERMSD, ERMSDCG
from pysages.methods import HistogramLogger, Unbiased

step_size = 2 * unit.femtosecond
nsteps = int(1e2)

pdb = app.PDBFile("../../../inputs/GAGA.box_0mM.pdb")
basetype2selectedAA = {
    "A": ["C8", "N6", "C2"],
    "U": ["C6", "O4", "O2"],
    "G": ["C8", "O6", "N2"],
    "C": ["C6", "N4", "O2"],
}

B123_indices = []

for i, res in enumerate(pdb.topology.residues()):
    if res.name in basetype2selectedAA.keys():
        B123_residue = dict.fromkeys(basetype2selectedAA[res.name])
        for atom in res.atoms():
            if atom.name in basetype2selectedAA[res.name]:
                B123_residue[atom.name] = atom.index
    B123_indices.append(B123_residue)

# notice that the order of the indices for eRMSD is tricky!
B123_indices_ordered = []
for i, res in enumerate(pdb.topology.residues()):
    if res.name in basetype2selectedAA.keys():
        B123 = B123_indices[i]
        B123_indices_ordered.extend((B123[Bname] for Bname in basetype2selectedAA[res.name]))

reference_CG = pdb.getPositions(asNumpy=True).astype("float")[np.asarray(B123_indices_ordered)]

sequence = [res.name for res in pdb.topology.residues() if res.name in "AUGC"]
nt2idx = {nt: idx for nt, idx in zip(["A", "U", "G", "C"], [0, 1, 2, 3])}
sequence = [nt2idx[s] for s in sequence]

C246_indices = []
for i, res in enumerate(pdb.topology.residues()):
    C246_residue = dict.fromkeys(["C2", "C4", "C6"])
    for atom in res.atoms():
        if atom.name in ["C2", "C4", "C6"]:
            C246_residue[atom.name] = atom.index
    C246_indices.append(C246_residue)

# notice that the order of the indices for eRMSD is tricky!
C246_indices_ordered = []
for i, res in enumerate(pdb.topology.residues()):
    if res.name in ["G", "A"]:
        C246 = C246_indices[i]
        C246_indices_ordered.extend((C246["C2"], C246["C6"], C246["C4"]))
    elif res.name in ["U", "C"]:
        C246 = C246_indices[i]
        C246_indices_ordered.extend((C246["C2"], C246["C4"], C246["C6"]))

reference = pdb.getPositions(asNumpy=True).astype("float")[np.asarray(C246_indices_ordered)]


def generate_simulation():
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometer,
        constraints=app.HBonds,
    )

    integrator = mm.LangevinIntegrator(
        298 * unit.kelvin, 5 / unit.picosecond, step_size.in_units_of(unit.picosecond)
    )
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    print("minimizing energy...")
    simulation.minimizeEnergy()

    print("Using {} platform".format(simulation.context.getPlatform().getName()))
    simulation.reporters.append(app.DCDReporter("output.dcd", 1, enforcePeriodicBox=False))

    return simulation


def main():
    cvs = [
        ERMSDCG(B123_indices_ordered, reference_CG, sequence, cutoff=3.2),
        ERMSD(C246_indices_ordered, reference, cutoff=3.2),
    ]

    method = Unbiased(cvs, jit_compile=False)
    callback = HistogramLogger(1)

    raw_result = pysages.run(method, generate_simulation, nsteps, callback)
    np.savetxt("ermsd_cg.txt", raw_result.callbacks[0].data[:, :1])
    np.savetxt("ermsd.txt", raw_result.callbacks[0].data[:, 1:2])


if __name__ == "__main__":
    main()
