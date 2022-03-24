PySAGES Examples
================

We provide two types of examples: Google Colab notebooks, which allow for a fast and easy introduction to PySAGES,
and python scripts that are ready to be run on your machine/cluster with your own installation.
If you are starting using PySAGES, we invite you to test and modify them for your needs.
The Google Colab notebooks can be used with Google's GPUs in case you do not have easy access to one.

## HOOMD-blue

Examples for Methods using HOOMD-blue can be found in the subfolder [hoomd-blue](hoomd-blue)

### Notebooks

 * Harmonic bias restricting a particle in space: [![Harmonic Bias](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/Harmonic_Bias.ipynb)
 * Umbella integration of a particle in an external potential: [![Umbrella Integation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/Umbrella_Integration.ipynb)
 * Artificial neural networks sampling of the dihedral angle of Butane: [![ANN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/Butane_ANN.ipynb)

### Script examples

* Harmonic bias restricting a particle in space: [harmonic_bias](hoomd-blue/harmonic_bias)
* Umbella integration of a particle in an external potential: [umbrella_integration](hoomd-blue/umbrella_integration)

## OpenMM

Examples for Methods using OpenMM can be found in the subfolder [openmm](openmm)

### Notebooks

 * Harmonic bias for the dihedral angle of Alanine Dipeptide: [![Harmonic Bias](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/openmm/Harmonic_Bias.ipynb)

### Script examples

* ABF with Alanine Dipeptide: [ABF](openmm/abf)

## Installation of the Environment

We have a Google Colab that shows how the MD engines and PySAGES can be installed together as guide to install PySAGES.
This notebook has a final step that sets up precompiled environments for all the other notebooks as well.
[![Install Env](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Install_PySAGES_Environment.ipynb)

## For Developers

If you add a method or a CV please try to add an example for your addition here as well.
Feel free to use any of the existing examples as templates or create your own. A script example or Google Colab works, ideally, you add both, but be sure to not include unnecessary binary data for your example i.e. try to generate initial conditions instead of shipping them.

We are also aiming to run the examples with the CI testing, so if you add your example and add it to the CI testing as well. If your example needs longer run time (we do not have GPUs for CI, yet) make sure to include an option to run it with fewer steps and or replica to finish in about 10 min.)

### Notebooks

All notebooks come in 2 flavors:

* The `.ipynb` should contain a full run-through and results of all cells.
* the `.md` is human readable and used for code revision.

It is our responsibility as developers to have versions in sync with one another in the beginning.
We recommend to use the `jupytext` package for this. Once a PR is submitted, we have a pre-commit hook to keep them synchronized.

