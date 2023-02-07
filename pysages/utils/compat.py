# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from importlib import import_module

import jax
import jaxlib
import jaxlib.xla_extension as xe
from jax.scipy import linalg

# Compatibility utils


def try_import(new_name, old_name):
    try:
        return import_module(new_name)
    except ModuleNotFoundError:
        return import_module(old_name)


def _version_as_tuple(ver_str):
    return tuple(int(i) for i in ver_str.split(".") if i.isdigit())


# Compatibility for jax >=0.4.1

# https://github.com/google/jax/releases/tag/jax-v0.4.1
if _version_as_tuple(jaxlib.__version__) < (0, 4, 1):
    JaxArray = xe.DeviceArray

    def check_device_array(array):
        pass

else:
    JaxArray = jax.Array

    def check_device_array(array):
        if not (array.is_fully_addressable and len(array.sharding.device_set) == 1):
            err = "Support for SharedDeviceArray or GlobalDeviceArray has not been implemented"
            raise ValueError(err)


# Compatibility for jax >=0.3.15

# https://github.com/google/jax/compare/jaxlib-v0.3.14...jax-v0.3.15
# https://github.com/google/jax/pull/11546
if _version_as_tuple(jaxlib.__version__) < (0, 3, 15):

    def solve_pos_def(a, b):
        return linalg.solve(a, b, sym_pos="sym")

else:

    def solve_pos_def(a, b):
        return linalg.solve(a, b, assume_a="pos")
