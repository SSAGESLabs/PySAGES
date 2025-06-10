# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# flake8: noqa F401
# pylint: disable=unused-import,relative-beyond-top-level

"""
Advanced Sampling Methods
=========================

Advanced sampling methods are summarized in this sub-module.
Methods have two objectives in PySAGES.

- Building python functions that bias simulations.
- Conducting the simulations run. This can include a single replica run, but also the
  orchestration of multiple replicas with complex interactions between one another.

The biasing part is implemented, such that each class provides a
:py:meth:`.core.SamplingMethod.build` member function.
This function is called internally by PySAGES once at initialization and returns two
functions.

- an initialize function to generate the first internal state for biasing.
- and an update function, that invokes the calculation of the biasing forces.

This functional design is mandated by the :py:mod:`jax` implementation of PySAGES.
The functions are just in time compiled for maximum performance.
For new methods, the user has to implement this interface for custom biasing.

The conducting of simulation runs is designed closer to python's object-oriented design.
The :py:meth:`core.SamplingMethod.run` function uses a user-provided function to generate
the simulation context for the chosen backend.
This member function sets up the necessary replica (simple ones only need one) of the
simulation, conducts the bias simulation. Depending on the methods it may also collect
information for analysis.

Each method inherits an abstract base implementation from SamplingMethod, see for details
the class documentation.
Any non-abstract method class has an accompanying state.
This state is a data class for JAX and can contain only JAXArray of fixed dimensions.
Methods use this state to carry information to conduct their biasing.
There are two special members each state should provide:

- :py:attr:`bias` is an array of shape `(Nparticles, 3)` which must contain the biasing
  forces for each particle after the invocation of the biasing function.
- :py:attr:`xi` contains the last state of the collective variables used for biasing.

More members are allowed to provide the necessary information.

Biasing and simulation orchestration can be separated into a different classes.
The :py:class:`harmonic_bias.HarmonicBias` class for example provides a
:py:meth:`harmonic_bias.HarmonicBias.build` function for generate functions for harmonic
biasing forces.
The methods, however, just inherit the basic implementation of a single replica run.
:py:class:`umbrella_integration.UmbrellaIntegration` on the other hand, does not implement
a new biasing method (and thus has no internal state as well).
Instead it inherits the biasing from :py:class:`harmonic_bias.HarmonicBias` but
re-implements :py:meth:`umbrella_integration.UmbrellaIntegration.run`
to sample multiple replicas along a path to estimate free energy differences.
"""

from .abf import ABF
from .ann import ANN
from .bias import Bias
from .cff import CFF
from .core import SamplingMethod
from .ffs import FFS
from .funn import FUNN
from .funnel_function import get_funnel_force
from .funnel_metad import Funnel_Metadynamics
from .funnel_sabf import Funnel_SpectralABF
from .harmonic_bias import HarmonicBias
from .metad import Metadynamics
from .nanoreactor import Nanoreactor
from .old_fabf import Funnel_ABF
from .restraints import CVRestraints
from .sirens import Sirens
from .spectral_abf import SpectralABF
from .spline_string import SplineString
from .umbrella_integration import UmbrellaIntegration
from .unbiased import Unbiased
from .utils import (
    Funnel_Logger,
    Funnel_MetadLogger,
    HistogramLogger,
    MetaDLogger,
    ReplicasConfiguration,
    SerialExecutor,
    methods_dispatch,
)
