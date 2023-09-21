# Installation

The dependencies for a PySAGES installation can change based on your desired simulation
backend engine. Depending on your system setup, it may be necessary to reinstall the
simulation backend, especially if you do not have write permission for your current
installation. On top of the current installation, it is required that you install a
plugin that connects PySAGES with the simulation engine:

- [DLPack Plugin for HOOMD-blue](https://github.com/SSAGESLabs/hoomd-dlext)
- [DLPack Plugin for OpenMM](https://github.com/SSAGESLabs/openmm-dlext)
- [DLPack Plugin for LAMMPS](https://github.com/SSAGESLabs/lammps-dlext)
- No plugin needed for ASE

You also need to install [JAX](https://github.com/google/jax), a library for
high-performance numerical computing. Follow their installation guide and make sure you
have it installed before using PySAGES. Depending on your local setup, you may need
to install the CPU version or the CUDA compatible version.

For GPU support, JAX and any backend (HOOMD-blue, LAMMPS, or OpenMM) need to be built or
installed with CUDA support.

Once the installation requirements are satisfied, PySAGES can be installed with `pip`:

```shell
pip install git+https://github.com/SSAGESLabs/PySAGES.git
```

or

```shell
git clone https://github.com/SSAGESLabs/PySAGES.git
cd PySAGES
pip install .
```

For the latest version of PySAGES, it is possible to deploy simulations with a [docker
container](https://hub.docker.com/r/ssages/pysages). However, at the moment we cannot
guarantee a stable docker image for PySAGES.

You can follow our installation tutorial on Google Colab to see how PySAGES, HOOMD-blue,
and OpenMM can be built and installed in such an environment: [Install Notebook].

<!-- References -->

[Install Notebook]: https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Install_PySAGES_Environment.ipynb
