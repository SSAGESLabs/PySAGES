# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import ctypes

import numba
import numpy
from numpy.ctypeslib import as_ctypes_type

from pysages.typing import JaxArray
from pysages.utils import dispatch


@dispatch
def view(array: JaxArray):
    """Return a writable view of a JAX DeviceArray."""
    # NOTE: We need a more general strategy to handle
    # `SharedDeviceArray`s and `GlobalDeviceArray`s.
    ptype = ctypes.POINTER(as_ctypes_type(array.dtype))
    addr = array.device_buffer.unsafe_buffer_pointer()
    ptr = ctypes.cast(ctypes.c_void_p(addr), ptype)
    return numba.carray(ptr, array.shape)


@dispatch
def view(array: numpy.ndarray):  # noqa: F811 # pylint: disable=C0116,E0102
    """Return a writable view of a numpy.ndarray."""
    ptype = ctypes.POINTER(as_ctypes_type(array.dtype))
    addr = array.__array_interface__["data"][0]
    ptr = ctypes.cast(ctypes.c_void_p(addr), ptype)
    return numba.carray(ptr, array.shape)
