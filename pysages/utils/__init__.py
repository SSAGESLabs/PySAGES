# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# flake8: noqa F401
# pylint: disable=unused-import,relative-beyond-top-level

"""
Miscellaneous utils for working with PySAGES.
"""

from .compat import (
    check_device_array,
    device_platform,
    dispatch_table,
    has_method,
    is_generic_subclass,
    prod,
    solve_pos_def,
    try_import,
    unsafe_buffer_pointer,
)
from .core import (
    ToCPU,
    copy,
    dispatch,
    eps,
    first_or_all,
    gaussian,
    identity,
    is_file,
    last,
    linear_solver,
    parse_array,
    splitlines,
)
from .transformations import quaternion_from_euler, quaternion_matrix
