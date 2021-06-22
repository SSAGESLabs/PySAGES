# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jax import scipy
import jax.numpy as np
try:
    import mpi4jax
except ImportError as exception:
    raise RuntimeError("The string methods requires MPI, mpi4py, and mpi4jax as dependencies. Please install them before installing PySAGES.") from exception


def interpolate_cv(s_out, comm, cv=None, token=None, root_rank=0, norm_function=None):
    if norm_function is None:
        norm_function = lambda a, b: np.sqrt(np.mean((a-b)**2))

    if comm.Get_rank() == root_rank:
        if cv is None:
            raise RuntimeError("Passing a cv on rank {0} (root) is non-optinal.".format(root_rank))

    # Validate correct cv shape
    dims = 0
    if cv is not None:
        dims = len(cv.shape)
    dims, token = mpi4jax.bcast(np.asarray(dims, dtype=int), root_rank, comm=comm, token=token)
    cv_shape = np.zeros(dims, dtype=int)
    if cv is not None:
        if len(cv.shape) != dims:
            raise RuntimeError("Incompatible dimensions for CV found on rank "+str(comm.Get_rank()))
        cv_shape = np.asarray(cv.shape, dtype=int)
    cv_shape, token = mpi4jax.bcast(cv_shape, root_rank, comm=comm, token=token)
    if cv is not None:
        if not np.array_equal(cv_shape, np.asarray(cv.shape, dtype=int)):
            raise RuntimeError("Incompatible shapes for CV found on rank "+str(comm.Get_rank()))
    cv_shape = tuple(cv_shape)

    # Recv all CVs on the root rank
    if comm.Get_rank() != root_rank:
        has_data = np.asarray(cv is not None)
        token = mpi4jax.send(has_data, root_rank, comm=comm, token=token)
        if has_data:
            token = mpi4jax.send(cv, root_rank, comm=comm, token=token)
    else:  # ROOT
        cv_list = []
        for i in range(comm.Get_size()):
            if i == root_rank:
                cv_list.append(cv)
            else:
                has_data = np.asarray(False)
                has_data, token = mpi4jax.recv(has_data, i, comm=comm, token=token)
                if has_data:
                    other_cv, token = mpi4jax.recv(cv, i, comm=comm, token=token)
                    cv_list.append(other_cv)
        # create norm for the received cvs
        s_list = [0]
        for i in range(1, len(cv_list)):
            norm = norm_function(cv_list[i-1], cv_list[i])
            s_list.append(s_list[-1]+norm)
        s_list = np.asarray(s_list)
        s_list /= s_list[-1]

    # assemble the new cvs to be send out
    requested_s, token = mpi4jax.gather(float(s_out), 0, comm=comm, token=token)
    if comm.Get_rank() == root_rank:
        requested_cvs = []
        index_list = [np.zeros(1)]
        for i in range(len(cv_shape)):
            index_list.append(np.arange(cv_shape[i]))
        coordinates = list(np.meshgrid(*index_list))
        for i in range(len(coordinates)):
            coordinates[i] = coordinates[i].reshape(-1)

        cv_list = np.stack(cv_list)
        for i in range(comm.Get_size()):
            for j in range(len(s_list)-1):
                if requested_s[i] >= s_list[j] and requested_s[i] <= s_list[j+1]:
                    delta_s = s_list[j+1] - s_list[j]
                    s_distance = j + (requested_s[i] - s_list[j])/delta_s
                    break
            coordinates[0] = coordinates[0]*0 + s_distance
            new_cv = scipy.ndimage.map_coordinates(cv_list, coordinates, order=1, mode="nearest")
            new_cv = new_cv.reshape(cv_shape)
            print(new_cv.shape)
            requested_cvs.append(new_cv)
        requested_cvs = np.stack(requested_cvs)
        print(requested_cvs.shape)

        out_cv, token = mpi4jax.scatter(requested_cvs, root_rank, comm=comm, token=token)
    else:
        print(cv_shape)
        out_cv, token = mpi4jax.scatter(np.zeros(cv_shape), root_rank, comm=comm, token=token)

    return out_cv
