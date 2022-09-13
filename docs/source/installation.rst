Installation
============

The dependencies for a PySAGES installation depend on your desired simulation backend engine.
Depending on your system setup, it may be necessary to reinstall the simulation backend, especially if you do not have write permission for your current installation.
On top of the current installation, it is required that you install a plugin that connects PySAGES with the simulation engine:

 - For HOOMD-blue, visit `HOOMD DLPack Plugin <https://github.com/SSAGESLabs/hoomd-dlext>`__
 - For OpenMM, go to `OpenMM DLPack Plugin <https://github.com/SSAGESLabs/openmm-dlext>`__

and follow their installation instructions.

PySAGES also depends on `JAX <https://github.com/google/jax/>`__; follow their installation
guide to set it up. *NOTE:* make sure you have jaxlib installed before using PySAGES.
Depending on your local setup, you will have to install the jaxlib CPU version or the CUDA-compatible flavor.
To utilize full GPU with PySAGES CUDA support of HOOMD-blue, HOOMD-dlext and JAX are required.

Once the installation requirements are satisfied, PySAGES can be installed with `pip`.::

   pip install git+https://github.com/SSAGESLabs/PySAGES.git

or

   git clone https://github.com/SSAGESLabs/PySAGES.git
   cd PySAGES
   pip install .
For the latest version of PySAGES, it is possible to deploy simulations with a `docker container <https://hub.docker.com/r/ssages/pysages>`__.
At the moment we cannot guarantee a stable docker image for PySAGES.

Our installation tutorial on Google Colab enables you to see how PySAGES,
HOOMD-blue, and OpenMM can be built and installed:
`Install Colab <https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Install_PySAGES_Environment.ipynb>`__
