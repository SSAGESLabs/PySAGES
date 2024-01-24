# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# flake8: noqa F401
# pylint: disable=unused-import,relative-beyond-top-level

"""
Miscellaneous utils for working with PySAGES.
"""

from .compat import (
    check_device_array,
    dispatch_table,
    has_method,
    is_generic_subclass,
    solve_pos_def,
    try_import,
)
from .core import ToCPU, copy, dispatch, eps, gaussian, identity, only_or_identity
from .transformations import quaternion_from_euler, quaternion_matrix
