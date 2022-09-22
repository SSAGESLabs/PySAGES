# PySAGES

[![Documentation Status](https://readthedocs.org/projects/pysages/badge/?version=latest)](https://pysages.readthedocs.io/en/latest/?badge=latest)
[![GitHub Actions](https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml)
[![GitHub Actions](https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml/badge.svg?branch=main)](https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml)
[![GitHub Actions](https://github.com/SSAGESLabs/PySAGES/actions/workflows/trunk.yml/badge.svg?branch=main)](https://github.com/SSAGESLabs/PySAGES/actions/workflows/trunk.yml)

PySAGES (Python Suite for Advanced General Ensemble Simulations) is a Python
implementation of [SSAGES](https://ssagesproject.github.io) with support for GPUs.

**NOTICE**: This is in early stages of development. Expect breaking changes.

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
guide to set it up. _NOTE:_ make sure you have jaxlib installed before using PySAGES.
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

## Development

We believe in good software engineering and collaboration.
As an open-source software we welcome all contributions.
To ease collaboration we use [trunk](https://trunk.io) as a development tool free for open-source software.
This includes version-checked linters that can be run automatically.
We ship a launcher for trunk with this repo `./trunk`, no installation required.
For details about how to use `./trunk fmt` to format existing code into this style and `./trunk check` to verify a consistent code style, check out the trunk documentation [page](https://docs.trunk.io/docs).

### Trunk githooks

For the development of this repo, we highly recommend to activate trunk's interaction with githooks.
The ensures, that any code committed and pushed fulfills the coding standards. (Otherwise a CI test will fail on the repo side PRs.)
This feature requires to be activated by developers on each machine.

```shell
./trunk git-hooks install
```
