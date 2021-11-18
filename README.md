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

To test GPU support HOOMD-blue, HOOMD-dlext and JAX need to be built or installed with
GPU support.

## Usage

Let's update

PySAGES provide a straightforward interface to setup Collective Variables and Enhaced
Sampling methods to you MD simulations. See the documentation to learn more.

Here is an example of how to hook pysages to an OpenMM simulation of alanine dipeptide

```python
# Your code to define a OpenMM goes first

# Load pysages
from numpy import pi
import pysages

# Declare a list of collective variables
cvs = (
    pysages.collective_variables.DihedralAngle((4, 6, 8, 14)),
	pysages.collective_variables.DihedralAngle((6, 8, 14, 16))
)

# Define a grid if the sampling method requires one
grid = pysages.Grid(
    lower = (-pi, -pi),
    upper = ( pi,  pi),
    shape = (32, 32),
    periodic = True
)

# Declare the sampling method to be used
sampling_method = pysages.methods.ABF(ξ, grid, N = 100)

# Hook pysages to the simulation
sampler = pysages.bind(context, sampling_method)

# Now run the simulation for any number of time steps
```
