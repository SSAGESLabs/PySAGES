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
[![Install Env](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/docs/examples/Install_PySAGES_Environment.ipynb)

## Usage

PySAGES provide a straightforward interface to setup Collective Variables and Enhanced
Sampling methods to your MD simulations. See the [documentation](https://pysages.readthedocs.io/en/latest/) to learn more.

We are also offering ready to go examples for common methods.
Checkout our [examples](examples/) subfolder to learn about Google Colab examples and script examples.
The examples include simulations and an example of how to install PySAGES with the MD engines.
