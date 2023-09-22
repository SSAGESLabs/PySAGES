<!--
You can adapt this file completely to your liking, but it should contain
the root ```{toctree}``` block.
-->

# Python Suite for Advanced General Ensemble Simulations

<!-- markdownlint-disable MD053 -->

````{grid} 1
:margin: 4 4 0 0
:padding: 0

```{grid-item}
[![CI Badge]](https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml)  &nbsp;
[![Docker CI Badge]](https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml)  &nbsp;
[![Trunk Badge]](https://github.com/SSAGESLabs/PySAGES/actions/workflows/trunk.yml)
```
````

Molecular dynamics (MD) simulations are powerful tools to investigate the static and
dynamic properties of a given system. However, even with modern computer architecture and
the fastest simulation software, computation time is limited and valuable. As a result,
exploring a system by unbiased MD is insufficient to obtain good statistics, especially if
the free-energy landscape is separated by high barriers. To still investigate systems
with high energy barriers, enhanced-sampling methods have been established. Typically, a
configuration can be reduced to a collective variable (order parameter), and the
simulation is biased based on these collective variables. The challenge for computer
simulations is that _i_&hairsp;) almost every interesting system has its own collective
variable description and _ii_&hairsp;) the implementation of collective variables and
methods has to run efficiently on modern computers, to allow reasonable insights into the
observable of interest.

```{rubric} No compromises in usability and speed for enhanced-sampling methods
:class: sd-fs-5
```

PySAGES addresses these challenges by offering a python interface between highly optimized
simulation engines and the researcher to implement collective variables and
enhanced-sampling methods. If you are new to advanced sampling techniques, you can try out
this [interactive introduction][Intro] with PySAGES. Even better, PySAGES already provides
an extensible framework to compute collective variables and to perform enhanced-sampling
MD simulations to discover reaction pathways and estimate free energies. Most research
objectives are achievable by using these implemented collective variables and methods.
PySAGES currently supports automatically connecting these methods to [ASE], [HOOMD-blue],
[LAMMPS], and [OpenMM]. PySAGES communicates with these backends without copying data
between GPU and host memory, except for ASE, which does not support GPU calculations
directly. This approach allows biased simulations without slowing the backend simulation
engines down. PySAGES still implements all methods and collective variables in pure python
for access and modification.

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`

```{toctree}
:caption: Contents
:maxdepth: 2
:hidden:

getting-started
installation
trouble-shooting
mpi
package-pysages
```

<!-- References -->

[CI Badge]: https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml/badge.svg?branch=main
[Docker CI Badge]: https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml/badge.svg?branch=main
[Trunk Badge]: https://github.com/SSAGESLabs/PySAGES/actions/workflows/trunk.yml/badge.svg?branch=main
[Intro]: https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Advanced_Sampling_Introduction.ipynb
[ASE]: https://wiki.fysik.dtu.dk/ase/index.html
[HOOMD-blue]: https://glotzerlab.engin.umich.edu/hoomd-blue
[LAMMPS]: https://www.lammps.org
[OpenMM]: https://openmm.org
