# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jaxlib.xla_extension import DeviceArray
from numpy.ctypeslib import as_ctypes_type
from plum import dispatch

import ctypes
import numba
import numpy


@dispatch
def view(array: DeviceArray):
    """Return a writable view of a JAX DeviceArray."""
    ptype = ctypes.POINTER(as_ctypes_type(array.dtype))
    addr = array.device_buffer.unsafe_buffer_pointer()
    ptr = ctypes.cast(ctypes.c_void_p(addr), ptype)
    return numba.carray(ptr, array.shape)


@dispatch
def view(array: numpy.ndarray):
    """Return a writable view of a numpy.ndarray."""
    ptype = ctypes.POINTER(as_ctypes_type(array.dtype))
    addr = array.__array_interface__["data"][0]
    ptr = ctypes.cast(ctypes.c_void_p(addr), ptype)
    return numba.carray(ptr, array.shape)
