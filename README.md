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
dynamics engine:

 - For HOOMD-blue visit [HOOMD-dlpack Plugin](https://github.com/SSAGESLabs/hoomd-dlext).
 - For OpenMM go to [OpenMM](https://github.com/SSAGESLabs/openmm-dlext).

PySAGES also depends on [JAX](https://github.com/google/jax/), follow their installation
guide to set it up. *NOTE:* make sure you manually install jaxlib before PySAGES.
Depending on your local setup, you will have to install the jaxlib CPU version or the CUDA compatible flavor.

To test GPU support HOOMD-blue, HOOMD-dlext and JAX need to be built or installed with
GPU support.

Our tutorials on google colab enable you to see how PySAGES can be installed into an environment that supports these engines.
[![Hoomd-blue Harmonic Bias](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/docs/notebooks/Harmonic_Bias_PySAGES_HOOMD.ipynb)

## Usage

PySAGES provide a straightforward interface to setup Collective Variables and Enhanced
Sampling methods to your MD simulations. See the documentation to learn more.

If you want to learn more, you can also head over to the `examples` subfolder,
which contains scripts for running different HOOMD-blue and OpenMM biased simulations.

Or checkout one of our google colab notebooks to try out PySAGES yourself.

### HOOMD-blue

Simple Harmonic Biasing:

[![Hoomd-blue Harmonic Bias](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/docs/notebooks/Harmonic_Bias_PySAGES_HOOMD.ipynb)

Umbrella Integration with harmonic bias:

[![Hoomd-blue Umbrella Integration](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/docs/notebooks/Umbrella_Integration_PySAGES_HOOMD.ipynb)


### OpenMM

Simple Harmonic Biasing:

[![OpenMM Harmonic Bias](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/docs/notebooks/Harmonic_Bias_PySAGES_OpenMM.ipynb)


### Installation of the environment

All of the previous notebooks require a preinstalled environment of OpenMM and HOOMD-blue with the plugings to hook them up to PySAGES.
As a reference for the installation instructions of these environments take a look at this notebook:

[![Install Env](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/docs/notebooks/Install_PySAGES_Environment.ipynb)
