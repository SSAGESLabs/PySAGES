# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
PySAGES: Python Suite for Advanced General Ensemble Simulations
"""

import os

# Check for user set memory environment for XLA/JAX
if not (
    "XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ
    or "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ
    or "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ
):
    # If not set be user, disable preallocate to enable multiple/growing
    # simulation memory footprints
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from ._version import (
    version as __version__,
    version_tuple,
)

from .backends import (
    ContextWrapper,
    supported_backends,
)

from .grids import (
    Chebyshev,
    Grid,
)

from .utils import (
    dispatch,
)

from . import (
    collective_variables,
    methods,
)

run = dispatch._functions["run"]
analyze = dispatch._functions["analyze"]
