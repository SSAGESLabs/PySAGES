# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from collections import namedtuple
from jax import scipy
import jax.numpy as np
import mpi4jax

def interpolate_cv(s_out, comm, cv=None, token=None, root_rank=0, norm_function=None):
    if norm_function is None:
        norm_function = lambda a, b: np.sqrt(np.mean((a-b)**2))

    # Send cv to root rank.
    if comm.Get_Rank() != root_rank:
        has_data = np.asarray(cv is None)
        token = mpi4jax.send(has_data, root_rank, comm=comm, token=token)
        if cv is not None:
            token = mpi4jax.send(cv, root_rank, comm=comm, token=token)
            cv_shape = cv.shape
        else:
            dims, token = mpi4jax.recv(np.zeros(1, dtype=int), root_rank, comm=comm, token=token)
            cv_shape, token = mpi4jax.recv(np.zeros(dims, dtype=int), root_rank, comm=comm, token=token)
            cv_shape = tuple(cv_shape)

        # collect all cvs from the ranks (if available)
        cv_list = []
        for i in range(comm.Get_Size()):
            if i == root_rank:
                if cv is None:
                    raise RuntimeError("Passing a cv on rank {0} (root) is non-optinal.".format(root_rank))
                cv_list.append(cv)
            else:
                has_data = np.asarray(False)
                has_data, token = mpi4jax.recv(has_data, i, comm=comm, token=token)
                if has_data:
                    other_cv, token = mpi4jax.recv(cv, i, comm=comm, token=token)
                    cv_list.append(other_cv)
                else:
                    token = mpi4jax.send(np.asarray(len(cv.shape), dtype=int), i, comm=comm, token=token)
                    token = mpi4jax.send(np.asarray(cv.shape, dtype=int), i, comm=comm, token=token)

        # create norm for the received cvs
        s_list = [0]
        for i in range(1, len(cv_list)):
            norm = norm_function(cv_list[i-1], cv_list[i])
            s_list.append(s_list[-1]+norm)
        s_list = np.asarray(s_list)


    #assemble the new cvs to be send out
    requested_s, token = mpi4jax.gather(float(s_out), 0, comm=comm, token=token)
    if comm.Get_Rank() == root_rank:
        requesetd_cv = np.zeros((comm.Get_Size(),)+cv_shape)

        index_list = []

        cv_list = np.stack(cv_list)

        for i in range(comm.Get_Size()):
            s = requested_s[i]

            # the interpolation command that is implemented with jax. is probably scipy.ndimage.map_coordinates
            # but I can't think of a good way to genrate the coordinates, yet.
