.. PySAGES documentation master file, created by
   sphinx-quickstart on Tue Dec  7 14:02:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PySAGES: Python Suite for Advanced General Ensemble Simulations
===============================================================


.. only:: html

    |Github Actions CI|
    |Github Actions CI Docker|
    |Documentation Status|

    .. |Documentation Status| image:: https://readthedocs.org/projects/pysages/badge/?version=latest
        :target: https://pysages.readthedocs.io/en/latest/?badge=latest
    .. |GitHub Actions CI| image:: https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml/badge.svg?branch=main
        :target: https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml
    .. |GitHub Actions CI Docker| image:: https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml/badge.svg?branch=main
        :target: https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml


Molecular dynamics (MD) simulations are powerful tools to investigate the static and dynamic properties of a given system.
However, even with modern computer architecture and the fastest simulation software, computation time is limited and valuable.
As a result, exploring a system by unbiased MD is insufficient to obtain good statistics, especially if the free-energy landscape is separated by high barriers.
To still investigate systems with high energy barriers, enhanced-sampling methods have been established.
Typically, a configuration can be reduced to a collective variable (order parameter), and the simulation is biased based on these collective variables.
The challenge for computer simulations is that i) almost every interesting system has its own collective variable description and ii) the implementation
of collective variables and methods has run to efficiently on modern computers, to allow reasonable insights into the observable of interest.

No compromises in usability and speed for enhanced-sampling methods!
--------------------------------------------------------------------

PySAGES addresses these challenges by offering a python interface between highly optimized simulation engines and the researcher to implement collective variables and enhanced-sampling methods.
Even better, PySAGES already provides an extensible framework to compute collective variables and to perform enhanced-sampling MD simulations to discover reaction pathways and estimate free energies.
Most research objectives are achievable by using these implemented collective variables and methods.
PySAGES currently supports automatically connecting these methods to `HOOMD-blue <https://glotzerlab.engin.umich.edu/hoomd-blue>`__ and
`OpenMM <http://openmm.org/>`__.
Both engines offer a python interface to the user and implement the simulation on GPUs for best performance.
PySAGES interacts with both backends directly on the GPU memory; copying between GPU and host memory is not required.
This approach allows biased simulations without slowing the backend simulation engines down.
PySAGES still implements all methods and collective variables as pure python for access and modification.


Getting Started
---------------

PySAGES is designed to seamlessly integrate with your existing simulations.
For an approachable start we prepared examples in Google Colab documents, which provide easy access to a working simulation environment and GPU hardware.
For an example of using PySAGES with HOOMD-blue, check out `this <https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/Umbrella_Integration.ipynb>`__ example for simple umbrella integration.
If you are working with OpenMM, we offer `this <https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/openmm/Harmonic_Bias.ipynb>`__ notebook, which explores the free energy landscape of the conformational space of alanine dipeptide via harmonic biasing.
Using the notebooks enables you to get started with PySAGES without installation.
For a productive deployment, we recommend a full installation.

We offer more examples, in the form of both Google Colab notebooks and for fully installed PySAGES, in the GitHub repository `examples <https://github.com/SSAGESLabs/PySAGES/tree/main/examples>`__.


Installation
------------

The dependencies for a PySAGES installation depend on your desired simulation backend engine.
Depending on your system setup, it may be necessary to reinstall the simulation backend, especially if you do not have write permission for your current installation.
On top of the current installation, it is required that you install a plugin that connects PySAGES with the simulation engine:

 - For HOOMD-blue, visit `HOOMD DLPack Plugin <https://github.com/SSAGESLabs/hoomd-dlext>`__.
 - For OpenMM, go to `OpenMM DLPack Plugin <https://github.com/SSAGESLabs/openmm-dlext>`__.

And follow their installation instructions.

PySAGES also depends on `JAX <https://github.com/google/jax/>`__; follow their installation
guide to set it up. *NOTE:* make sure you have jaxlib installed before using PySAGES.
Depending on your local setup, you will have to install the jaxlib CPU version or the CUDA-compatible flavor.
To utilize full GPU with PySAGES CUDA support of HOOMD-blue, HOOMD-dlext and JAX are required.

Once the installation requirements are satisfied, PySAGES can be installed with `pip`.
```
git clone https://github.com/SSAGESLabs/PySAGES.git
cd PySAGES
pip install .
```

For the latest version of PySAGES, it is possible to deploy simulations with a `docker container <https://hub.docker.com/r/ssages/pysages>`__.
At the moment we cannot guarantee a stable docker image for PySAGES.

Our installation tutorial on Google Colab enables you to see how PySAGES,
HOOMD-blue, and OpenMM can be built and installed:
`Install Colab <https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Install_PySAGES_Environment.ipynb>`__


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   package-pysages

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
