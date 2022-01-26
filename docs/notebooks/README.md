PySAGES Google Colab Jupyter Notebooks
======================================

PySAGES' comes with a set of Jupyter notebooks designed to make it easy for users to get started.
All of them are designed to run on Google's colab service for easy acces to GPUs.

All notebooks come in 2 flavors:
* the .ipynb should contain a full run-through and results of all cells.
* the .md is human readbable and used for code revision.

It is our responsibilyt as developers to have versions in sync with one another. The `jupytext` tool is a great help for that.

## Installation of the Environment

Since these notebooks require HOOMD-blue and OpenMM to be installed, we provide a pre-compiled enviroment that all notebooks use.
The generation of this environment can be seen in `Install_PySAGES_Environment.*` but ask the core maintainers if you need to update the environment.