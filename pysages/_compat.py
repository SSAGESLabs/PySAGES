# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# flake8: noqa F401
# pylint: disable=unused-import,relative-beyond-top-level

from importlib import import_module
from platform import python_version

from plum._version import __version_tuple__ as _plum_version_tuple

# Compatibility utils


def _version_as_tuple(ver_str):
    return tuple(int(i) for i in ver_str.split(".") if i.isdigit())


_jax_version_tuple = _version_as_tuple(import_module("jaxlib").__version__)
_python_version_tuple = _version_as_tuple(python_version())
