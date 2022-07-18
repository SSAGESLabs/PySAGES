# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# flake8: noqa F401

"""
PySAGES: Python Suite for Advanced General Ensemble Simulations
"""

import os


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


_set_cuda_visible_devices()
del _set_cuda_visible_devices

# Check for user set memory environment for XLA/JAX
if not (
    "XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ
    or "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ
    or "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ
):
    # If not set be user, disable preallocate to enable multiple/growing
    # simulation memory footprints
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from ._version import (  # noqa: E402, F401
    version as __version__,
    version_tuple,
)

from .backends import (  # noqa: E402, F401
    ContextWrapper,
    supported_backends,
)

from .grids import (  # noqa: E402, F401
    Chebyshev,
    Grid,
)

from .methods import (  # noqa: E402, F401
    CVRestraints,
    ReplicasConfiguration,
    SerialExecutor,
)

from .utils import (  # noqa: E402, F401
    dispatch,
)

from . import (  # noqa: E402, F401
    colvars,
    methods,
)

run = dispatch._functions["run"]
analyze = dispatch._functions["analyze"]
