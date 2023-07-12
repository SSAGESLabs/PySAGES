# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# pylint: disable=unused-import,relative-beyond-top-level
# flake8: noqa F401

"""
Collective variables describe a macro state with many possible micro state realizations.
PySAGES tries to support a variety common collective variables, but it is easy to extend
PySAGES with your own.
"""

from .angles import Angle, DihedralAngle
from .coordinates import Component, Displacement, Distance
from .shape import (
    Acylindricity,
    Asphericity,
    PrincipalMoment,
    RadiusOfGyration,
    ShapeAnisotropy,
)
from .utils import get_periods, wrap

# Conditionally export GeM if both `jax_md` and `jaxopt` are available
try:
    import jax_md
    import jaxopt

    from .patterns import GeM
except ImportError:
    pass
