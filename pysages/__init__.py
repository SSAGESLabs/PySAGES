# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# flake8: noqa E402,F401

"""
PySAGES: Python Suite for Advanced General Ensemble Simulations
"""

import os

import jax


def _set_cuda_visible_devices():
    """
    Determine the local MPI rank and CUDA_VISIBLE_DEVICES.
    Assign the GPU to the local rank.

    The assumptions here is that every node has the same number of GPUs available.
    """
    local_mpi_rank = None
    try:  # OpenMPI
        local_mpi_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    except KeyError:
        pass
    try:
        local_mpi_rank = int(os.environ["MV2_COMM_WORLD_LOCAL_RANK"])
    except KeyError:
        pass

    passed_visible_devices = None
    try:
        passed_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    except KeyError:
        pass

    if local_mpi_rank and passed_visible_devices:
        gpu_num_id = local_mpi_rank % len(passed_visible_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = passed_visible_devices[gpu_num_id]


def _config_jax():
    # Check for user set memory environment for XLA/JAX
    if not (
        "XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ
        or "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ
        or "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ
    ):
        # If not set be user, disable preallocate to enable multiple/growing
        # simulation memory footprints
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Set default floating point type for arrays in `jax` to `jax.f64`
    jax.config.update("jax_enable_x64", True)


_set_cuda_visible_devices()
_config_jax()

# pylint: disable=C0413
from . import backends, colvars, methods
from ._version import version as __version__
from ._version import version_tuple as __version_tuple__
from .backends import supported_backends
from .grids import Chebyshev, Grid
from .methods import CVRestraints, ReplicasConfiguration, SerialExecutor
from .serialization import load, save
from .utils import dispatch, dispatch_table

run = dispatch_table(dispatch)["run"]
analyze = dispatch_table(dispatch)["analyze"]


# Reduce namespace noise
del dispatch_table
del jax
del os
del _config_jax
del _set_cuda_visible_devices
del _version  # pylint: disable=E0602
