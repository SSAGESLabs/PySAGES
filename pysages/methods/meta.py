# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import Callable, Mapping, NamedTuple, Optional

from pysages.backends import ContextWrapper
from pysages.methods.core import SamplingMethod, generalize
from pysages.utils import JaxArray, copy

# from pysages.grids import build_indexer

import jax.numpy as np
from jax import lax
from jax import jit

from itertools import product

# ================ #
#   Metadynamics   #
# ================ #

# =======================================================  #
# =======================================================  #
# Log
#   v0: standard meta without grids and single CV
#   v1: standard meta without grids and multiple CVs
#   v2: standard meta with grids and multiple CVs
#   v3: well-tempered meta without grids and multiple CVs
#   v4: well-tempered meta with grids and multiple CVs
#   v5: support restart using hills file - pending
# =======================================================  #
# =======================================================  #


# callback to log hills and other output files
class logMeta:
    """
    Implements a Callback functor for methods.
    Logs the state of the collective variable and other parameters in metadynamics.
    """

    def __init__(
        self, hills_period, colvar_period, sigma, height, hillsFile, colvarFile, grid_defined, angle
    ):
        """
        logMeta constructor.

        Arguments
        ---------
        hills_period:
            Timesteps between logging of collective variables and metadynamics parameters.

        colvar_period:
            Timesteps between logging of collective variables and bias potential.

        sigma:
            Width of the Gaussian bias potential.

        height:
            Height of the Gaussian bias potential.

        hillsFile:
            Name of the output hills log file.

        colvarFile:
            Name of the output colvar log file.
        """
        self.hills_period = hills_period
        self.colvar_period = colvar_period
        self.sigma = sigma
        self.height = height
        self.hillsFile = hillsFile
        self.colvarFile = colvarFile
        self.grid_defined = grid_defined
        self.angle = angle
        self.counter = 0

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """

        if self.counter != 0 and self.counter % self.hills_period == 0:

            write_hills_to_file(
                self.counter,
                state.xi_stride[state.loop - 1],
                state.sigma_stride[state.loop - 1],
                state.height_stride[state.loop - 1],
                self.hillsFile,
            )

        # to do -- track bias potential when calculating bias forces for efficiency.
        if self.counter % self.colvar_period == 0:

            # NOTE: state.xi is the next timestep; implies hills and colvar will not match.
            # this should not be a problem because colvar is primarily for data analysis.
            # for consistency, timestep in colvar is incremented.
            # Potential issue: bias_pot from meta_bias_pot will not correspond to state.xi (colvar_nstrides?)

            if self.grid_defined and self.hills_period != self.colvar_period:
                # compute bias potentials again because with grids grid_potential is updated only at hills_period
                bias_pot = 0
                bias_pot = lax.fori_loop(
                    0,
                    state.loop - 1,
                    bias_potential,
                    (
                        bias_pot,
                        state.xi,
                        state.xi_stride,
                        state.sigma_stride,
                        state.height_stride,
                        self.angle,
                    ),
                )[0]
            else:
                bias_pot = state.meta_bias_pot[self.counter][0]

            write_colvar_to_file(self.counter + 1, state.xi, bias_pot, self.colvarFile)

        self.counter += 1


class MetaBiasState(NamedTuple):

    """
    Description of bias by metadynamics bias potential for a CV.

    bias -- Array of metadynamics bias forces for each particle in the simulation.
    xi -- Collective variable value in the last simulation step.
    xi_stride -- Collective variable value at the last stride.
    sigma_stride -- sigma value at the last stride.
    height_stride -- height value at the last stride.
    loop -- tracking strides during the simulation for filling xi_stride.
    grid_force -- Array of metadynamics bias forces for each particle in the simulation stored on a grid.
    """

    bias: JaxArray
    xi: JaxArray

    xi_stride: JaxArray
    sigma_stride: JaxArray
    height_stride: JaxArray

    loop: int
    frame: int

    meta_bias_pot: JaxArray

    grid_force: JaxArray
    grid_potential: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class meta(SamplingMethod):
    def __init__(self, cvs, height, sigma, stride, nstrides, deltaT, *args, **kwargs):

        super().__init__(cvs, args, kwargs)

        self.height = height
        self.sigma = sigma
        self.stride = stride
        self.nstrides = nstrides
        self.deltaT = deltaT

        # to do: remove this, if PySAGES can extract kB from engine (snapshot.kB)
        if "kB" in kwargs:
            self.kB = kwargs["kB"]
        else:
            self.kB = 8.314462618 * 1e-3  # default

        if "colvar_outstride" in kwargs:
            self.colvar_outstride = kwargs["colvar_outstride"]
        else:
            self.colvar_outstride = 100  # default

        if "hillsFile" in kwargs:
            self.hillsFile = kwargs["hillsFile"]
        else:
            self.hillsFile = "hills.dat"  # default

        if "colvarFile" in kwargs:
            self.colvarFile = kwargs["colvarFile"]
        else:
            self.colvarFile = "colvar.dat"  # default

        if "grid" in kwargs:
            self.grid = kwargs["grid"]

        if "angle" in kwargs:
            self.angle = kwargs["angle"]
        else:
            self.angle = np.full(np.shape(cvs)[0], False)

        self.snapshot_flags = ("positions", "indices")

    def build(self, snapshot, helpers):

        self.helpers = helpers

        return metadynamics(self, snapshot, helpers)

    # Overriding default run for meta
    def run(
        self,
        context_generator: Callable,
        timesteps: int,
        verbose: bool = False,
        callback: Optional[Callable] = None,
        context_args: Mapping = dict(),
        **kwargs
    ):
        """
        Metadynamics algorithm.

        Arguments
        ---------
        context_generator: Callable
            User defined function that sets up a simulation context with the backend.
            Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
            and `simtk.openmm.Simulation` for OpenMM. The function gets `context_args`
            unpacked for additional user arguments.

        timesteps: int
            Number of timesteps the simulation is running.

        verbose: bool
            If True more information will be logged (useful for debbuging).

        callback: Optional[Callable]
            Allows for user defined actions into the simulation workflow of the method.
            `kwargs` gets passed to the backend `run` function.

        """

        grid_defined = False
        if hasattr(self, "grid"):

            grid_defined = True
            grid = self.grid
            dims = grid.shape.size

            # to do -- may be move to grids.py
            # get_grid_index = build_indexer(grid)
            lower_edges, upper_edges, grid_spacing = build_edges(grid, grid.shape)

            iterList = []
            n_grid_points = 1
            for d in range(dims):
                iterList.append(range(grid.shape[d]))
                n_grid_points *= grid.shape[d]

            grid_force = np.zeros((*grid.shape, dims), dtype=np.float64)

            grid_potential = np.zeros((*grid.shape, 1), dtype=np.float64)

            # to do - use indices from grid_force for generating coordinates
            grid_indices = np.asarray(np.meshgrid(*np.asarray(iterList)))
            grid_indices = np.stack((*lax.map(np.ravel, grid_indices),)).T

        context = context_generator(**context_args)
        callback = logMeta(
            self.stride,
            self.colvar_outstride,
            self.sigma,
            self.height,
            self.hillsFile,
            self.colvarFile,
            grid_defined,
            self.angle,
        )
        wrapped_context = ContextWrapper(context, self, callback)

        with wrapped_context:

            sampler = wrapped_context.sampler
            run = wrapped_context.run

            loop = 0
            for frame in range(0, timesteps):

                # print(f"Frame: {frame}\n")

                if (frame) != 0 and (frame) % self.stride == 0:

                    xi_stride = sampler.state.xi_stride
                    for j in range(sampler.state.xi.shape[1]):
                        xi_stride = xi_stride.at[(loop, j)].set(sampler.state.xi[0][j])

                    height_stride = sampler.state.height_stride

                    # loop += 1

                    if grid_defined:

                        if self.deltaT > 0:

                            # read grid potential
                            index_xi = jit(get_grid_index)(
                                sampler.state.xi, lower_edges, grid_spacing, grid.shape
                            )
                            total_bias_potential = sampler.state.grid_potential[index_xi][0] / (
                                self.deltaT * self.kB
                            )
                            height_stride = height_stride.at[loop].set(
                                self.height * np.exp(-total_bias_potential)
                            )

                        # loop through each grid point and update grid_force and grid_potential
                        grid_result = lax.fori_loop(
                            0,
                            n_grid_points,
                            derivative_bias_grid,
                            (
                                sampler.state.grid_force,
                                sampler.state.grid_potential,
                                grid_indices,
                                lower_edges,
                                grid_spacing,
                                sampler.state.xi,
                                sampler.state.sigma_stride[loop],
                                height_stride[loop],
                                self.angle,
                            ),
                        )
                        grid_force, grid_potential, _, _, _, _, _, _, _ = grid_result

                    else:

                        grid_force = None
                        grid_potential = None

                        # modify height_stride if using well-tempered meta
                        if self.deltaT > 0:
                            print("deltaT")
                            print(self.deltaT)
                            print("\n")
                            total_bias_potential = 0
                            total_bias_potential = lax.fori_loop(
                                0,
                                loop,
                                bias_potential_wmeta,
                                (
                                    total_bias_potential,
                                    sampler.state.xi,
                                    xi_stride,
                                    sampler.state.sigma_stride,
                                    self.height,
                                    self.deltaT * self.kB,
                                    self.angle,
                                ),
                            )[0]
                            total_bias_potential = total_bias_potential / (self.deltaT * self.kB)
                            height_stride = height_stride.at[loop].set(
                                self.height * np.exp(-total_bias_potential)
                            )

                    loop += 1

                    sampler.state = MetaBiasState(
                        sampler.state.bias,
                        sampler.state.xi,
                        xi_stride,
                        sampler.state.sigma_stride,
                        height_stride,
                        loop,
                        frame,
                        sampler.state.meta_bias_pot,
                        grid_force,
                        grid_potential,
                    )

                run(1, **kwargs)


def metadynamics(method, snapshot, helpers):

    cv = method.cv

    grid_defined = False
    if hasattr(method, "grid"):
        grid_defined = True
        grid = method.grid
        dims = grid.shape.size
        # get_grid_index = build_indexer(grid)
        lower_edges, upper_edges, grid_spacing = build_edges(grid, grid.shape)

        n_grid_points = 1
        for d in range(dims):
            n_grid_points *= grid.shape[d]

    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)

    height = method.height
    sigma = method.sigma
    stride = method.stride
    nstrides = method.nstrides
    deltaT = method.deltaT

    colvar_outstride = method.colvar_outstride
    hillsFile = method.hillsFile
    colvarFile = method.colvarFile

    angle = method.angle

    # initialize method
    def initialize():

        bias = np.zeros((natoms, 3), dtype=np.float64)
        xi, _ = cv(helpers.query(snapshot))

        xi_stride = np.zeros((nstrides, xi.shape[1]), dtype=np.float64)
        sigma_stride = np.full((nstrides, xi.shape[1]), sigma, dtype=np.float64)
        height_stride = np.full((nstrides), height, dtype=np.float64)

        loop = 0
        frame = 0

        # NOTE: bias potentials need not be store every frame;
        # reqd may be only at timesteps//colvar_outstride + 1
        meta_bias_pot = np.zeros((stride * nstrides, 1), dtype=np.float64)

        if grid_defined:
            grid_force = np.zeros((*grid.shape, dims), dtype=np.float64)
            grid_potential = np.zeros((*grid.shape, 1), dtype=np.float64)
        else:
            grid_force = None
            grid_potential = None

        return MetaBiasState(
            bias,
            xi,
            xi_stride,
            sigma_stride,
            height_stride,
            loop,
            frame,
            meta_bias_pot,
            grid_force,
            grid_potential,
        )

    def update(state, data):

        # calculate CV and its gradient dCV/dr
        xi, Jxi = cv(data)

        bias = state.bias

        # calculate gradient bias potential along CV
        dbias_dxi = np.zeros(state.xi.shape[1], dtype=np.float64)

        # change this to lax cond. to do
        if grid_defined:

            # look up and apply dbias_dxi and bias potential using grids
            index_xi = jit(get_grid_index)(xi, lower_edges, grid_spacing, grid.shape)
            dbias_dxi = state.grid_force[index_xi]
            meta_bias_pot = state.grid_potential[index_xi]

        else:

            bias_result = lax.fori_loop(
                0,
                state.loop,
                derivative_bias,
                (
                    dbias_dxi,
                    state.meta_bias_pot[state.frame][0],
                    xi,
                    state.xi_stride,
                    state.sigma_stride,
                    state.height_stride,
                    angle,
                ),
            )
            dbias_dxi, meta_bias_pot, _, _, _, _, _ = bias_result

        updated_meta_bias_pot = state.meta_bias_pot
        updated_meta_bias_pot = updated_meta_bias_pot.at[state.frame].set(meta_bias_pot)

        # calculate forces
        bias = -Jxi.T @ dbias_dxi.flatten()
        bias = bias.reshape(state.bias.shape)

        return MetaBiasState(
            bias,
            xi,
            state.xi_stride,
            state.sigma_stride,
            state.height_stride,
            state.loop,
            state.frame,
            updated_meta_bias_pot,
            state.grid_force,
            state.grid_potential,
        )

    return snapshot, initialize, generalize(update, helpers, jit_compile=False)


## HELPER FUNCTIONS

# expect arrays as inputs and return array as output
def exp_in_gaussian(delta_xi, sigma):

    sigma_square = 2 * np.multiply(sigma, sigma)
    delta_xi_square = np.multiply(delta_xi, delta_xi)
    arg = np.divide(delta_xi_square, sigma_square)

    return np.exp(-arg)


# expect floats as inputs and return float as output
def derivative_exp_in_gaussian(delta_xi, sigma):

    arg = (delta_xi * delta_xi) / (2 * sigma * sigma)

    pre = -(delta_xi) / (sigma * sigma)

    return pre * np.exp(-arg)


# for multiple CVs
def dbias_dxi_cv(j, dbias_dxi_cv_parameters):

    dbias_dxi, xi, xi_stride, sigma_cv, angle = dbias_dxi_cv_parameters

    delta_xi = xi - xi_stride

    # print("delta xi shape")
    # print(delta_xi.shape)

    delta_xi = periodic_angle(delta_xi, angle)

    sigma = sigma_cv

    exp_in_gaussian_local = exp_in_gaussian(delta_xi, sigma)
    exp_in_gaussian_local = exp_in_gaussian_local.at[j].set(1)

    ncvs_product = np.product(exp_in_gaussian_local, axis=0)

    local_delta_xi = delta_xi[j]  # xi[j] - xi_stride[j]

    # print("delta xi next shape")
    # print(local_delta_xi.shape)
    # print("end")
    # delta_xi = np.expand_dims(delta_xi, 0)
    # local_angle = np.expand_dims(angle[0][j], 0)
    # print(delta_xi.shape)
    # delta_xi = lax.fori_loop(0, delta_xi.shape[0], periodic_angle, (delta_xi, local_angle))[0]
    # delta_xi = np.squeeze(delta_xi, 0)
    # print(delta_xi.shape)
    # print("\n")

    sigma = sigma_cv[j]

    ncvs_product *= derivative_exp_in_gaussian(local_delta_xi, sigma)

    dbias_dxi = dbias_dxi.at[j].set(ncvs_product)

    return (dbias_dxi, xi, xi_stride, sigma_cv, angle)


# to do: add bias potential calculation
def derivative_bias(i, dbias_parameters):

    dbias_dxi, bias_pot, xi, xi_stride, sigma_stride, height_stride, angle = dbias_parameters

    delta_xi = xi[0] - xi_stride[i]

    # print("delta deriv shape")
    # print(delta_xi.shape)
    # print(delta_xi.shape)
    delta_xi = periodic_angle(delta_xi, angle)
    # print(delta_xi.shape)
    # print("end deriv 1\n")

    height = height_stride[i]

    sigma_cv = sigma_stride[i]

    dbias_dxi_local = dbias_dxi
    dbias_dxi_local = lax.fori_loop(
        0, xi.shape[1], dbias_dxi_cv, (dbias_dxi_local, xi[0], xi_stride[i], sigma_cv, angle)
    )[0]

    dbias_dxi = dbias_dxi + height * dbias_dxi_local

    bias_pot = bias_pot + height * np.product(exp_in_gaussian(delta_xi, sigma_cv), axis=0)

    return (dbias_dxi, bias_pot, xi, xi_stride, sigma_stride, height_stride, angle)


# bias forces and potential on a grid
def derivative_bias_grid(i, dbias_parameters):

    (
        grid_dbias_dxi,
        grid_bias,
        grid_index,
        lower_edges,
        grid_spacing,
        xi,
        sigma,
        height,
        angle,
    ) = dbias_parameters

    grid_index_i = jit(get_jit_index)(grid_index[i])

    coordinate = jit(get_grid_coordinates)(np.asarray(grid_index_i), lower_edges, grid_spacing)

    sigma_local = sigma
    dbias_dxi_local = grid_dbias_dxi[grid_index_i]
    dbias_dxi_local = lax.fori_loop(
        0, xi.shape[1], dbias_dxi_cv, (dbias_dxi_local, coordinate, xi[0], sigma_local, angle)
    )[0]

    grid_dbias_dxi = grid_dbias_dxi.at[grid_index_i].add(height * dbias_dxi_local)

    delta_xi = coordinate - xi[0]

    # print("dsad shape")
    # print(delta_xi.shape)

    delta_xi = periodic_angle(delta_xi, angle)
    # print("end this")

    grid_bias = grid_bias.at[grid_index_i].add(
        height * np.product(exp_in_gaussian(delta_xi, sigma), axis=0)
    )

    return (
        grid_dbias_dxi,
        grid_bias,
        grid_index,
        lower_edges,
        grid_spacing,
        xi,
        sigma,
        height,
        angle,
    )


def bias_potential(i, bias_parameters):

    bias_pot, xi, xi_stride, sigma_stride, height_stride, angle = bias_parameters

    delta_xi = xi[0] - xi_stride[i]

    # print("\n")
    # print("delta biax shape")
    # print(delta_xi.shape)
    # print(delta_xi.shape)
    delta_xi = periodic_angle(delta_xi, angle)
    # print(delta_xi.shape)
    # print("end \n")

    height = height_stride[i]

    sigma = sigma_stride[i]

    bias_pot = bias_pot + height * np.product(exp_in_gaussian(delta_xi, sigma), axis=0)

    return (bias_pot, xi, xi_stride, sigma_stride, height_stride, angle)


def bias_potential_wmeta(i, bias_parameters):

    bias_pot, xi, xi_stride, sigma_stride, height_initial, deltaT_kB, angle = bias_parameters

    delta_xi = xi[0] - xi_stride[i]

    delta_xi = periodic_angle(delta_xi, angle)

    height = height_initial * np.exp(-bias_pot / deltaT_kB)

    sigma = sigma_stride[i]

    bias_pot = bias_pot + height * np.product(exp_in_gaussian(delta_xi, sigma), axis=0)

    return (bias_pot, xi, xi_stride, sigma_stride, height_initial, deltaT_kB, angle)


def write_hills_to_file(frame, xi, sigma, height, hillsFile):

    with open(hillsFile, "a+") as f:

        f.write(str(frame) + "\t")
        for j in range(xi.shape[0]):

            f.write(str(xi[j]) + "\t")

        for j in range(sigma.shape[0]):

            f.write(str(sigma[j]) + "\t")

        f.write(str(height) + "\n")


def write_colvar_to_file(frame, xi, bias_potential, colvarFile):

    with open(colvarFile, "a+") as f:

        f.write(str(frame) + "\t")
        for j in range(xi.shape[1]):

            f.write(str(xi[0][j]) + "\t")

        f.write(str(bias_potential) + "\n")


def get_grid_coordinates(grid_index, lower_edges, grid_spacing):

    coordinate = lower_edges + np.multiply(grid_index + 0.5, grid_spacing)

    return coordinate


def build_edges(grid, num_points):

    lower_edges = grid.lower
    upper_edges = grid.upper

    grid_spacing = np.divide(upper_edges - lower_edges, num_points - 1)

    lower_edges -= grid_spacing / 2
    upper_edges += grid_spacing / 2

    return (lower_edges, upper_edges, grid_spacing)


def get_jit_index(idx):

    return (*np.flip(np.uint32(idx)),)


def get_grid_index(x, lower_grid, grid_spacing, grid_shape):

    idx = np.rint((x[0] - lower_grid) / grid_spacing - 0.5)

    # currently always periodic -- to do for both conditions
    idx = idx % (grid_shape - 1)

    # trigger out of bounds error?
    idx = lax.select(np.all(idx < np.full(idx.shape, 0)), np.full(idx.shape, np.nan), idx)
    idx = lax.select(np.all(idx > grid_shape), np.full(idx.shape, np.nan), idx)

    return (*np.uint32(idx),)


def periodic_angle(cvdifference, angle):

    # return cv difference taking periodicity into account

    # non periodidc angle or any other non periodic CV
    def nonperiodic_case(cvdiff):

        return cvdiff

    # periodic angle - wraps angle between -pi and pi
    # periodic CVs other than angle - not supported
    def periodic_angle_case(cvdiff):
        def neg_case(cvdiff):
            cvdiff = cvdiff + np.float64(2.0 * np.pi)
            return cvdiff

        def pos_case(cvdiff):
            cvdiff = cvdiff - np.float64(2.0 * np.pi)
            return cvdiff

        cvdiff = lax.cond(np.all(cvdiff > np.pi), pos_case, nonperiodic_case, cvdiff)
        cvdiff = lax.cond(np.all(cvdiff < -np.pi), neg_case, nonperiodic_case, cvdiff)

        return cvdiff

    def select_case(i, case_params):

        local_cvdifference, local_angle = case_params

        updated_cvdifference = lax.cond(
            np.all(local_angle[i]), periodic_angle_case, nonperiodic_case, local_cvdifference[i]
        )

        local_cvdifference = local_cvdifference.at[i].set(updated_cvdifference)

        return local_cvdifference, local_angle

    cvdifference, angle = lax.fori_loop(0, angle.shape[0], select_case, (cvdifference, angle))

    return cvdifference
