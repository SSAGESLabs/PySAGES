# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
This module defines the types to used for dispatch or as type hints.
"""

from importlib import import_module

import jax

from pysages._compat import _jax_version_tuple, _plum_version_tuple

# Compatibility for jax >=0.4.1

# https://github.com/google/jax/releases/tag/jax-v0.4.1
if _jax_version_tuple < (0, 4, 1):
    xe = import_module("jaxlib.xla_extension")
    JaxArray = xe.DeviceArray
    del xe
else:
    JaxArray = jax.Array


# Compatibility for plum >=2

# Since plum-dispatch>=2 depends on beartype, we need to be aware of
# https://beartype.readthedocs.io/en/latest/api_roar/#pep-585-deprecations
if _plum_version_tuple < (2, 0, 0):
    _typing = import_module("typing")
else:
    _typing = import_module("beartype.typing")


# Typing aliases
Any = _typing.Any
Callable = _typing.Callable
List = _typing.List
Iterable = _typing.Iterable
NamedTuple = _typing.NamedTuple
Optional = _typing.Optional
Sequence = _typing.Sequence
Tuple = _typing.Tuple
Union = _typing.Union

# Union aliases
Scalar = Union[None, bool, int, float]


# Remove namespace noise
del jax
del import_module
del _jax_version_tuple
del _plum_version_tuple
