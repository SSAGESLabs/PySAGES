# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import Callable, Mapping, NamedTuple, Optional

from pysages.backends import ContextWrapper
from pysages.methods.core import SamplingMethod, generalize
from pysages.utils import JaxArray, copy

from pysages.grids import build_indexer

import jax.numpy as np
from jax import lax, jit

from itertools import product

# ================ #
#   Metadynamics   #
# ================ #

# ===================================================================================  #
# ===================================================================================  #
# Log
#   v0: standard meta without grids and single CV
#   v1: standard meta without grids and multiple CVs
#   v2: standard meta with grids and multiple CVs
#   v3: well-tempered meta without grids and multiple CVs
#   v4: well-tempered meta with grids and multiple CVs
#   v5: converted loops to functions wherever possible
#   v6: modified bias pot calculation for colvar, used pysages.grids and added docs
# ===================================================================================  #
# ===================================================================================  #


# callback to log hills and other output files
# NOTE: for OpenMM; issue #16 on openmm-dlext should be resolved for this to work properly.
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

        grid_defined:
            Check if grid is defined.

        angle:
            Check if CV is an angle.

        counter:
            Local frame counter.
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
        local_loop = lax.select(state.loop == 0, 0, state.loop - 1)
        # Write hills file containing CV centers and corresponding heights
        if self.counter != 0 and self.counter % self.hills_period == 0:

            write_hills_to_file(
                self.counter,
                state.xi_stride[local_loop],
                state.sigma_stride[local_loop],
                state.height_stride[local_loop],
                self.hillsFile,
            )

        # Write colvar file containing CVs and bias potentials at defined stride
        if self.counter % self.colvar_period == 0:

            # NOTE: state.xi is the next timestep;
            # this should not be a problem because colvar is primarily for fes estimation using reweight approach.
            # for consistency between "hills" xi and "colvar" xi below, timestep in colvar is incremented.
            # however bias_pot would be still different because "if" computes bias pot with current xi and
            # "else" extracts bias_pot corresponding to xi in previous step.
            if self.grid_defined and self.hills_period != self.colvar_period:
                # compute bias potentials again because with grids grid_potential is updated only at hills_period
                bias_pot = bias_potential(
                    local_loop,
                    state.xi,
                    state.xi_stride,
                    state.sigma_stride,
                    state.height_stride,
                    self.angle,
                )

            else:
                bias_pot = state.meta_bias_pot[self.counter][0]

            write_colvar_to_file(self.counter + 1, state.xi, bias_pot, self.colvarFile)

        self.counter += 1


class MetaBiasState(NamedTuple):

    """
    Description of bias by metadynamics bias potential for a CV.

    bias: Array of metadynamics bias forces for each particle in the simulation.

    xi: Collective variable value in the last simulation step.

    xi_stride: Collective variable value at the last stride.

    sigma_stride: sigma value at the last stride.

    height_stride: height value at the last stride.

    loop: tracking strides during the simulation for filling xi_stride.

    frame:  frame number for updating metadynamics bias potential.

    meta_bias_pot: Array of metadynamics bias potential.

    grid_force: Array of metadynamics bias forces for each particle in the simulation stored on a grid.

    grid_potential: Array of metadynamics bias potential stored on a grid.
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

    """
    Sampling method for metadynamics.
    """

    def __init__(self, cvs, height, sigma, stride, nstrides, deltaT, *args, **kwargs):

        """
        Description of method parameters.

        Arguments
        ---------

        cvs: Collective variable.

        height: Initial Gaussian height.

        sigma: Initial width of Gaussian.

        stride: Bias potential deposition stride.

        nstrides: Total number of expected strides (timesteps//stride + 1).

        deltaT: deltaT for well-tempered metadynamics (if < 0; standard metadynamics is applied).

        Keyword arguments
        -----------------

        kB: Boltzmann constant.

        colvar_outstride: frequency to write output colvar file.

        ncolvarstrides: Expected size of bias potential array to store (default -- timesteps).

        hillsFile: Output hills filename.

        colvarFile: Output colvar filename.

        grid: CV on a grid.

        angle: Boolean array of size equal to CV to indicate CV is angle.
        """

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

        if "ncolvarstrides" in kwargs:
            self.ncolvarstrides = kwargs["ncolvarstrides"]
        else:
            self.ncolvarstrides = self.nstrides * self.stride - self.stride  # default

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
            self.angle = np.full(np.shape(cvs)[0], False)  # default

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

        # Check if grid is defined
        grid_defined = False
        if hasattr(self, "grid"):
            # Prepare for using grids
            grid_defined = True
            grid_indices = construct_grid(self.grid.shape)
            lower_edges, upper_edges, grid_spacing = build_edges(
                self.grid.lower, self.grid.upper, self.grid.shape
            )
            get_grid_index = build_indexer(self.grid)

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
            # start simulation
            for frame in range(0, timesteps):

                # print(f"Frame: {frame}\n")

                # add bias at user defined stride
                if (frame) != 0 and (frame) % self.stride == 0:

                    # Updated storing CV centers
                    xi_stride = sampler.state.xi_stride
                    xi_stride = xi_stride.at[loop].set(sampler.state.xi[0])

                    height_stride = sampler.state.height_stride

                    # grid defined
                    if grid_defined:

                        # sigma is always same in standard and well-tempered.
                        # for generality, current sigma is extracted from array
                        current_sigma = sampler.state.sigma_stride[loop]

                        # current height
                        current_height = self.height

                        # update current height for well-tempered metadynamics
                        if self.deltaT > 0:

                            # update height
                            index_xi = get_grid_index(sampler.state.xi)
                            total_bias_potential = sampler.state.grid_potential[index_xi][0] / (
                                self.deltaT * self.kB
                            )
                            current_height = jit(update_height)(total_bias_potential, self.height)
                            height_stride = height_stride.at[loop].set(current_height)

                        # update forces and potential stored on the grid
                        grid_force = sampler.state.grid_force
                        grid_potential = sampler.state.grid_potential

                        # calculate force and potential on grid
                        grid_params = (
                            grid_indices,
                            lower_edges,
                            grid_spacing,
                            sampler.state.xi,
                            current_sigma,
                            current_height,
                            self.angle,
                        )
                        _grid_force, _grid_potential = jit(derivative_bias_grid)(grid_params)

                        grid_force = jit(np.add)(
                            grid_force, _grid_force.reshape(sampler.state.grid_force.shape)
                        )
                        grid_potential = jit(np.add)(
                            grid_potential,
                            _grid_potential.reshape(sampler.state.grid_potential.shape),
                        )

                    # non grid approach
                    else:

                        grid_force = None
                        grid_potential = None

                        # modify height_stride if using well-tempered meta
                        if self.deltaT > 0:

                            # fori implementation [loop implementation is difficult to avoid here]
                            # in this case, height in the current stride depends on bias; which depends on previous height.
                            # also, lax.fori_loop implementation performance seems similar to that of non loop implementation
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

                    # update state
                    sampler.state = MetaBiasState(
                        sampler.state.bias,
                        sampler.state.xi,
                        xi_stride,
                        sampler.state.sigma_stride,
                        height_stride,
                        loop,
                        sampler.state.frame,
                        sampler.state.meta_bias_pot,
                        grid_force,
                        grid_potential,
                    )

                # run one timestep
                run(1, **kwargs)


def metadynamics(method, snapshot, helpers):

    """
    Initialization and update of bias forces. Interface as expected for methods.

    """
    # to do: for restart; use hills file to initialize corresponding arrays.

    cv = method.cv

    grid_defined = False
    if hasattr(method, "grid"):
        grid_defined = True
        grid = method.grid
        dims = grid.shape.size

        get_grid_index = build_indexer(grid)
        lower_edges, upper_edges, grid_spacing = build_edges(grid.lower, grid.upper, grid.shape)

    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)

    height = method.height
    sigma = method.sigma
    stride = method.stride
    nstrides = method.nstrides
    deltaT = method.deltaT

    colvar_outstride = method.colvar_outstride
    ncolvarstrides = method.ncolvarstrides
    hillsFile = method.hillsFile
    colvarFile = method.colvarFile

    angle = method.angle

    # initialize method
    def initialize():

        # initialize bias forces and calculate initial CV
        bias = np.zeros((natoms, 3), dtype=np.float64)
        xi, _ = cv(helpers.query(snapshot))

        # initial arrays to store CV centers, sigma and height of Gaussians at the stride
        xi_stride = np.zeros((nstrides, xi.shape[1]), dtype=np.float64)
        sigma_stride = np.full((nstrides, xi.shape[1]), sigma, dtype=np.float64)
        height_stride = np.full((nstrides), height, dtype=np.float64)

        # local variables to track stride and frame number
        loop = 0
        frame = 0

        # initialize bias potential array for writing output colvarfile and well-tempered approach
        meta_bias_pot = np.zeros((ncolvarstrides, 1), dtype=np.float64)

        # initialize arrays to store forces and bias potential on a grid.
        if grid_defined:
            grid_force = np.zeros((*grid.shape + 1, dims), dtype=np.float64)
            grid_potential = np.zeros((*grid.shape + 1, 1), dtype=np.float64)
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

        # calculate CV and the gradient of bias potential along CV -- dBias/dxi
        xi, Jxi = cv(data)

        bias = state.bias

        # calculate gradient bias potential along CV
        dbias_dxi = np.zeros(state.xi.shape[1], dtype=np.float64)
        meta_bias_pot = state.meta_bias_pot

        # look up and apply dbias_dxi and bias potential using grids
        if grid_defined:

            index_xi = get_grid_index(xi)
            dbias_dxi = state.grid_force[index_xi]
            meta_bias_pot = state.grid_potential[index_xi]

        # calculate dbias_dxi and bias potential
        else:

            dbias_dxi, meta_bias_pot = derivative_bias(
                state.loop, xi, state.xi_stride, state.sigma_stride, state.height_stride, angle
            )

            dbias_dxi = lax.select(
                state.loop == 0, np.zeros(state.xi.shape[1], dtype=np.float64), dbias_dxi
            )
            meta_bias_pot = lax.select(state.loop == 0, np.float64(0.0), meta_bias_pot)

        # update bias potential array
        updated_meta_bias_pot = state.meta_bias_pot
        updated_meta_bias_pot = updated_meta_bias_pot.at[state.frame].set(meta_bias_pot)

        # calculate bias forces -- dxi/dr x dbias/dxi
        bias = -Jxi.T @ dbias_dxi.flatten()
        bias = bias.reshape(state.bias.shape)

        return MetaBiasState(
            bias,
            xi,
            state.xi_stride,
            state.sigma_stride,
            state.height_stride,
            state.loop,
            state.frame + 1,
            updated_meta_bias_pot,
            state.grid_force,
            state.grid_potential,
        )

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


## HELPER FUNCTIONS
##############################################################
# Output files related functions
##############################################################

# write hills file
def write_hills_to_file(frame, xi, sigma, height, hillsFile):

    with open(hillsFile, "a+") as f:

        f.write(str(frame) + "\t")
        for j in range(xi.shape[0]):

            f.write(str(xi[j]) + "\t")

        for j in range(sigma.shape[0]):

            f.write(str(sigma[j]) + "\t")

        f.write(str(height) + "\n")


# write colvar file
def write_colvar_to_file(frame, xi, bias_potential, colvarFile):

    with open(colvarFile, "a+") as f:

        f.write(str(frame) + "\t")
        for j in range(xi.shape[1]):

            f.write(str(xi[0][j]) + "\t")

        f.write(str(bias_potential) + "\n")


# for output colvar file
def bias_potential(loop, xi, xi_stride, sigma_stride, height_stride, angle):

    delta_xi = xi - xi_stride
    delta_xi = periodic_angle(delta_xi, angle)

    height = height_stride
    sigma = sigma_stride

    bias_pot = np.multiply(height, np.product(exp_in_gaussian(delta_xi, sigma), axis=1))
    mask = np.arange(bias_pot.shape[0]) < loop
    bias_pot = np.sum(bias_pot, axis=0, where=mask)

    return bias_pot


##############################################################
# grid related functions (may be move to grids.py)
##############################################################

# get coordinates of grid points
def get_grid_coordinates(grid_index, lower_edges, grid_spacing):

    coordinate = lower_edges + np.multiply(grid_index, grid_spacing)

    return np.flip(coordinate, axis=1)


# get grid spacing
def build_edges(lower_edges, upper_edges, grid_shape):

    grid_spacing = np.divide(upper_edges - lower_edges, grid_shape)

    return (lower_edges, upper_edges, grid_spacing)


# get indices for generating coordinates
def construct_grid(grid_shape):

    indicesArray = np.indices(grid_shape + 1)

    grid_indices = np.stack((*lax.map(np.ravel, indicesArray),)).T

    return grid_indices


##############################################################
# return cv difference taking periodicity into account
# may be move to angle CVs or remove when supported in it.
##############################################################
def periodic_angle(cvdifference, angle):

    # non periodidc angle or any other non periodic CV
    def nonperiodic_case(cvdiff):
        return cvdiff

    # periodic angle - wraps angle between -pi and pi
    # periodic CVs other than angle - not supported
    def neg_case(cvdiff):
        cvdiff = cvdiff + np.float64(2.0 * np.pi)
        return cvdiff

    def pos_case(cvdiff):
        cvdiff = cvdiff - np.float64(2.0 * np.pi)
        return cvdiff

    cvdifference = np.where(
        angle,
        (np.where(cvdifference > np.pi, pos_case(cvdifference), cvdifference)),
        nonperiodic_case(cvdifference),
    )

    cvdifference = np.where(
        angle,
        (np.where(cvdifference < -np.pi, neg_case(cvdifference), cvdifference)),
        nonperiodic_case(cvdifference),
    )

    return cvdifference


##############################################################
# bias potential and gradient along CV helper functions
##############################################################

# exponential term in Gaussian
def exp_in_gaussian(delta_xi, sigma):

    sigma_square = 2.0 * np.multiply(sigma, sigma)
    delta_xi_square = np.multiply(delta_xi, delta_xi)
    arg = np.divide(delta_xi_square, sigma_square)

    return np.exp(-arg)


# derivative of exponential term in Gaussian
def derivative_exp_in_gaussian(delta_xi, sigma):

    sigma_square = np.multiply(sigma, sigma)
    delta_xi_square = np.multiply(delta_xi, delta_xi)
    arg = np.divide(delta_xi_square, 2 * sigma_square)
    pre = np.divide(-delta_xi, sigma_square)

    return np.multiply(pre, np.exp(-arg))


# for multiple CVs (product of Gaussians) - shared with and without grids implementation
def dbias_dxi_ncvs(xi, xi_stride, sigma, angle):

    delta_xi = np.subtract(xi, xi_stride)
    delta_xi = periodic_angle(delta_xi, angle)

    exp_in_gaussian_local = exp_in_gaussian(delta_xi, sigma)
    exp_in_gaussian_local_product = np.product(exp_in_gaussian_local, axis=1)
    exp_in_gaussian_local_repeat = np.repeat(
        exp_in_gaussian_local_product, exp_in_gaussian_local.shape[1]
    )
    exp_in_gaussian_local_repeat = exp_in_gaussian_local_repeat.reshape(exp_in_gaussian_local.shape)

    ncvs_product = np.divide(exp_in_gaussian_local_repeat, exp_in_gaussian_local)
    ncvs_product = np.multiply(ncvs_product, derivative_exp_in_gaussian(delta_xi, sigma))

    return ncvs_product


# calculate bias forces and potential on a grid
def derivative_bias_grid(dbias_parameters):

    grid_index, lower_edges, grid_spacing, xi, sigma, height, angle = dbias_parameters

    coordinate = get_grid_coordinates(grid_index, lower_edges, grid_spacing)

    dbias_dxi_local = dbias_dxi_ncvs(coordinate, xi[0], sigma, angle)

    grid_dbias_dxi = np.multiply(height, dbias_dxi_local)

    delta_xi = np.subtract(coordinate, xi[0])
    delta_xi = periodic_angle(delta_xi, angle)

    grid_bias = np.multiply(height, np.product(exp_in_gaussian(delta_xi, sigma), axis=1))

    return (grid_dbias_dxi, grid_bias)


#  calculate bias forces and potential without grids
def derivative_bias(loop, xi, xi_stride, sigma_stride, height_stride, angle):

    height = height_stride
    sigma = sigma_stride

    dbias_dxi_local = dbias_dxi_ncvs(xi[0], xi_stride, sigma, angle)

    dbias_dxi = height.reshape(-1, 1) * dbias_dxi_local
    mask = np.arange(dbias_dxi.shape[0] * dbias_dxi.shape[1]) < loop * dbias_dxi.shape[1]
    mask = mask.reshape(-1, dbias_dxi.shape[1])
    dbias_dxi = np.sum(dbias_dxi, axis=0, where=mask)

    delta_xi = np.subtract(xi[0], xi_stride)
    delta_xi = periodic_angle(delta_xi, angle)

    bias_pot = np.multiply(height, np.product(exp_in_gaussian(delta_xi, sigma), axis=1))
    mask = np.arange(bias_pot.shape[0]) < loop
    bias_pot = np.sum(bias_pot, axis=0, where=mask)

    return (dbias_dxi, bias_pot)


# for updating heights in well-tempered metadynamics (with and without grids)
def update_height(total_bias_potential, height):

    current_height = height * np.exp(-total_bias_potential)

    return current_height


# for updating bias potential in well-tempered metadynamics (without grids)
def bias_potential_wmeta(i, bias_parameters):

    bias_pot, xi, xi_stride, sigma_stride, height_initial, deltaT_kB, angle = bias_parameters

    delta_xi = xi[0] - xi_stride[i]
    delta_xi = periodic_angle(delta_xi, angle)

    height = height_initial * np.exp(-bias_pot / deltaT_kB)
    sigma = sigma_stride[i]

    bias_pot = bias_pot + height * np.product(exp_in_gaussian(delta_xi, sigma), axis=0)

    return (bias_pot, xi, xi_stride, sigma_stride, height_initial, deltaT_kB, angle)
