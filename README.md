# PySAGES

[![GitHub Actions](https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml)
[![GitHub Actions](https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml/badge.svg?branch=main)](https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml)

PySAGES (Python Suite for Advanced General Ensemble Simulations) is an Python
implementation of [SSAGES](https://ssagesproject.github.io) with support for GPUs.

**NOTICE**: This is in early stages of development, expect breaking changes.

## Installation

Currently, there is only support for
[HOOMD-blue](https://glotzerlab.engin.umich.edu/hoomd-blue) and
[OpenMM](http://openmm.org/), but gradual support for other molecular dynamics engines is
planned (for instance, for all engines already supported by the original SSAGES).

You first need to install one of the following plugins depending on your molecular
dynamics engine:

 - For HOOMD-blue visit [HOOMD DLPack Plugin](https://github.com/SSAGESLabs/hoomd-dlext).
 - For OpenMM go to [OpenMM DLPack Plugin](https://github.com/SSAGESLabs/openmm-dlext).

PySAGES also depends on [JAX](https://github.com/google/jax/), follow their installation
guide to set it up. *NOTE:* make sure you have jaxlib installed before using PySAGES.
Depending on your local setup, you will have to install the jaxlib CPU version or the CUDA compatible flavor.

To test GPU support HOOMD-blue, HOOMD-dlext and JAX need to be built or installed with
CUDA support.

Our installation tutorial on Google Colab enable you to see how PySAGES,
HOOMD-blue and OpenMM can be built and installed into such environment.
[![Install Env](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Install_PySAGES_Environment.ipynb)

## Usage

PySAGES provide a straightforward interface to setup Collective Variables and Enhanced
Sampling methods in your MD simulations. See the [documentation](https://pysages.readthedocs.io/en/latest/) to learn more.

We provide ready-to-go examples for common methods.
Checkout out the [examples](examples/) subfolder to look at different script and notebook examples.
These include pre-set simulations and a tutorial on how to install PySAGES along with the supported MD engines.
