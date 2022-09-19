.. PySAGES documentation master file, created by
   sphinx-quickstart on Tue Dec  7 14:02:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PySAGES: Python Suite for Advanced General Ensemble Simulations
===============================================================


.. only:: html

    |Github Actions CI|
    |Github Actions CI Docker|
    |Github Actions Trunk|
    |Documentation Status|

    .. |Documentation Status| image:: https://readthedocs.org/projects/pysages/badge/?version=latest
        :target: https://pysages.readthedocs.io/en/latest/?badge=latest
    .. |GitHub Actions CI| image:: https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml/badge.svg?branch=main
        :target: https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml
    .. |GitHub Actions CI Docker| image:: https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml/badge.svg?branch=main
        :target: https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml
    .. |GitHub Actions Trunk| image:: https://github.com/SSAGESLabs/PySAGES/actions/workflows/trunk.yml/badge.svg?branch=main
        :target: https://github.com/SSAGESLabs/PySAGES/actions/workflows/trunk.yml


Molecular dynamics (MD) simulations are powerful tools to investigate the static and dynamic properties of a given system.
However, even with modern computer architecture and the fastest simulation software, computation time is limited and valuable.
As a result, exploring a system by unbiased MD is insufficient to obtain good statistics, especially if the free-energy landscape is separated by high barriers.
To still investigate systems with high energy barriers, enhanced-sampling methods have been established.
Typically, a configuration can be reduced to a collective variable (order parameter), and the simulation is biased based on these collective variables.
The challenge for computer simulations is that i) almost every interesting system has its own collective variable description and ii) the implementation
of collective variables and methods has to run efficiently on modern computers, to allow reasonable insights into the observable of interest.

No compromises in usability and speed for enhanced-sampling methods!
--------------------------------------------------------------------

PySAGES addresses these challenges by offering a python interface between highly optimized simulation engines and the researcher to implement collective variables and enhanced-sampling methods.
If you are new to advanced sampling techniques, you can try out this `interactive introduction <https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Advanced_Sampling_Introduction.ipynb>`__. with PySAGES.
Even better, PySAGES already provides an extensible framework to compute collective variables and to perform enhanced-sampling MD simulations to discover reaction pathways and estimate free energies.
Most research objectives are achievable by using these implemented collective variables and methods.
PySAGES currently supports automatically connecting these methods to `HOOMD-blue <https://glotzerlab.engin.umich.edu/hoomd-blue>`__ and
`OpenMM <http://openmm.org/>`__.
Both engines offer a python interface to the user and implement the simulation on GPUs for best performance.
PySAGES interacts with both backends directly on the GPU memory; copying between GPU and host memory is not required.
This approach allows biased simulations without slowing the backend simulation engines down.
PySAGES still implements all methods and collective variables as pure python for access and modification.




.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   installation
   trouble-shooting
   mpi
   package-pysages

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
