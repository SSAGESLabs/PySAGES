# PySAGES

PySAGES (Python Suite for Advanced General Ensemble Simulations) is an Python
implementation of [SSAGES](https://ssagesproject.github.io) with support for GPUs.

**NOTICE**: This is in early stages of development, expect breaking changes.

## Installation

Currently, there is only support for
[HOOMD-blue](https://glotzerlab.engin.umich.edu/hoomd-blue) and
[OpenMM](http://openmm.org/), but gradual support for other molecular dynamics engines is
planned (for instance, for all engines already supported by the original SSAGES).

You first need to install one of the following plugins depending on your molecular
dynamics engine

 - For HOOMD-blue visit [HOOMD-dlpack Plugin](https://github.com/SSAGESLabs/hoomd-dlext).
 - For OpenMM go to [OpenMM](https://github.com/SSAGESLabs/openmm-dlext).

PySAGES also depends on [JAX](https://github.com/google/jax/), follow their installation
guide to set it up.
It is important to install your flavor of jaxlib before you install PySAGES.
Make sure that you install the CUDA compatible flavor for your system or the CPU version, depending on your local setup.
PySAGES will not install jaxlib.

To test GPU support HOOMD-blue, HOOMD-dlext and JAX need to be built or installed with
GPU support.

## Usage

PySAGES provide a straightforward interface to setup Collective Variables and Enhaced
Sampling methods to you MD simulations. See the documentation to learn more.

Here is an example of how to hook pysages to an OpenMM simulation of alanine dipeptide

```python
from pysages.collective_variables import DihedralAngle
from pysages.methods import ABF
from pysages.utils import try_import

import numpy
import pysages

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")

pi = numpy.pi

# Your code to generate an OpenMM simulation generator.
# It is your normal OpenMM code wrap in a function.
def generate_simulation(
    pdb_filename = "alanine-dipeptide-explicit.pdb",
    T = 298.15 * unit.kelvin,
    dt = 2.0 * unit.femtoseconds
):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml", "tip3p.xml")
    cutoff_distance = 1.0 * unit.nanometer
    topology = pdb.topology

    system = ff.createSystem(
        topology, constraints = app.HBonds, nonbondedMethod = app.PME,
        nonbondedCutoff = cutoff_distance
    )

    # Set dispersion correction use.
    forces = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        forces[force.__class__.__name__] = force

    forces["NonbondedForce"].setUseDispersionCorrection(True)
    forces["NonbondedForce"].setEwaldErrorTolerance(1.0e-5)

    positions = pdb.getPositions(asNumpy = True)

    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)

    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    return simulation

# Declare a list of collective variables
cvs = (
    DihedralAngle((4, 6, 8, 14)),
    DihedralAngle((6, 8, 14, 16))
)
# Define a grid if the sampling method requires one
grid = pysages.Grid(
    lower = (-pi, -pi),
    upper = (pi, pi),
    shape = (32, 32),
    periodic = True
)

# Declare the sampling method to be used
method = ABF(cvs, grid)

# Now run the simulation for any number of time steps
method.run(generate_simulation, 50)
```
