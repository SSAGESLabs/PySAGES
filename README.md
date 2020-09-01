# PySAGES

PySAGES (Python Suite for Advanced General Ensemble Simulations) is an Python
implementation of [SSAGES](https://ssagesproject.github.io) with support for GPUs.

**NOTICE**: This is in early stages of development, expect failures and breaking changes.

## Installation

Currently, ther is only support for
[HOOMD-blue](https://glotzerlab.engin.umich.edu/hoomd-blue), but gradual support for other
molecular dynamics engines is planned (for instance, for all engines already supported by
the original SSAGES).

You need to first install HOOMD-blue and additionaly build the [HOOMD-dlpack
Plugin](https://github.com/pabloferz/hoomd-dlext).

PySAGES also depends on [JAX](https://github.com/google/jax/), follow their installation
guide to set it up.

To test GPU support HOOMD-blue, HOOMD-dlext and JAX need to be built or installed with
GPU support.
