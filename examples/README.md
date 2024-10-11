# PySAGES Examples

We provide two types of examples: Google Colab notebooks, which allow for a fast and easy introduction to PySAGES,
and python scripts that are ready to be run on your machine/cluster with your own installation.
If you are starting using PySAGES, we invite you to test and modify them for your needs.
The Google Colab notebooks can be used with Google's GPUs in case you do not have easy access to one.

If you are new to advanced sampling techniques in general you can try this interactive introduction with PySAGES [![Intro](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Advanced_Sampling_Introduction.ipynb).

## HOOMD-blue

Examples for Methods using HOOMD-blue can be found in the subfolders [hoomd-blue 2.x](hoomd-blue) and [hoomd-blue 3.x](hoomd3)

### HOOMD-blue notebooks

- Harmonic bias restricting a particle in space: [![Harmonic Bias](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/harmonic_bias/Harmonic_Bias.ipynb)
- Umbella integration of a particle in an external potential: [![Umbrella Integation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/umbrella_integration/Umbrella_Integration.ipynb)
- Artificial neural networks sampling of the dihedral angle of butane: [![ANN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/ann/Butane_ANN.ipynb)
- Adaptive force-biasing sampling of the dihedral angle of butane using neural networks: [![FUNN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/funn/Butane_FUNN.ipynb)
- Combined force-frequency sampling of the dihedral angle of butane: [![CFF](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/cff/Butane_CFF.ipynb)

### HOOMD-blue 2.x script examples

- Harmonic bias restricting a particle in space: [harmonic_bias](hoomd-blue/harmonic_bias)
- Umbrella integration of a particle in an external potential: [umbrella_integration](hoomd-blue/umbrella_integration)
- Artificial neural networks sampling of the dihedral angle of butane: [ann](hoomd-blue/ann/butane_ann.py)
- Adaptive force-biasing sampling of the dihedral angle of butane using neural networks: [funn](hoomd-blue/funn/butane.py)
- Combined force-frequency sampling of the dihedral angle of butane: [cff](hoomd-blue/cff/butane.py)
- Metadynamics sampling of the dihedral angle of butane: [metadynamics](hoomd-blue/metad/butane.py)
- Spline String sampling of a particle in an external potential: [spline-string](hoomd-blue/string/spline_string.py)

### HOOMD-blue 3.x script examples

- Harmonic bias restricting a particle in space: [harmonic_bias](hoomd3/harmonic_bias)
- Umbrella integration of a particle in an external potential: [umbrella_integration](hoomd3/umbrella_integration)
- Artificial neural networks sampling of the dihedral angle of butane: [ann](hoomd3/ann/butane_ann.py)
- Adaptive force-biasing sampling of the dihedral angle of butane using neural networks: [funn](hoomd3/funn/butane.py)
- Combined force-frequency sampling of the dihedral angle of butane: [cff](hoomd3/cff/butane.py)
- Metadynamics sampling of the dihedral angle of butane: [metadynamics](hoomd3/metad/butane.py)
- Spline String sampling of a particle in an external potential: [spline-string](hoomd3/string/spline_string.py)

## OpenMM

Examples for Methods using OpenMM can be found in the subfolder [openmm](openmm)

### OpenMM notebooks

- Harmonic bias for the dihedral angle of Alanine Dipeptide: [![Harmonic Bias](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/openmm/Harmonic_Bias.ipynb)
- Metadynamics sampling with Alanine Dipeptide: [![Metadynamics](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/openmm/metad/Metadynamics-ADP.ipynb)
- Metadynamics sampling with NaCL [![MetadynamicsNaCl](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/openmm/metad/nacl/Metadynamics_NaCl.ipynb)
- Spectral ABF sampling with Alanine Dipeptide: [![SpectralABF](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/openmm/spectral_abf/ADP_SpectralABF.ipynb)

### OpenMM script examples

- ABF with Alanine Dipeptide: [ABF](openmm/abf/alanine-dipeptide_openmm.py)
- Forward flux sampling with Alanine Dipeptide: [FFS](openmm/forward_flux_sampling.py)
- Metadynamics with Alanine Dipeptide: [Metadynamics](openmm/metad/alanine-dipeptide.py)
- Spectral ABF with Alanine Dipeptide: [SpectralABF](openmm/spectral_abf/alanine-dipeptide.py)
- Umbrella integration with Alanine Dipeptide: [Umbrella Integration](openmm/umbrella_integration/integration.py)

## Installation of the Environment

We have a Google Colab that shows how the MD engines and PySAGES can be installed together as a guide to install PySAGES.
This notebook has a final step that sets up precompiled environments for all the other notebooks as well.
[![Install Env](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Installing_a_PySAGES_Environment.ipynb)

## For Developers

If you add a method or a CV please try to add an example for your addition here as well.
Feel free to use any of the existing examples as templates or create your own. A script example or Google Colab works, ideally, you add both, but be sure to not include unnecessary binary data for your example i.e. try to generate initial conditions instead of shipping them.

We are also aiming to run the examples with the CI testing, so if you add your example and add it to the CI testing as well. If your example needs longer run time (we do not have GPUs for CI, yet) make sure to include an option to run it with fewer steps and or replica to finish in about 10 min.)

### Notebooks

All notebooks come in 2 flavors:

- The `.ipynb` should contain a full run-through and results of all cells.
- the `.md` is human readable and used for code revision.

It is our responsibility as developers to have versions in sync with one another in the beginning.
We recommend using the `jupytext` package for this. Once a PR is submitted, we have a pre-commit hook to keep them synchronized.
