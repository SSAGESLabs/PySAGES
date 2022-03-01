# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import Callable, Mapping, NamedTuple, Optional

from pysages.backends import ContextWrapper
from pysages.methods.core import SamplingMethod, generalize
from pysages.utils import JaxArray, copy

import jax.numpy as np
from jax import lax

# ================ #
#   Metadynamics   #
# ================ #

# =======================================================  #
# =======================================================  #
# Log
#   v0: standard meta without grids and single CV
#   v1: standard meta without grids and multiple CVs
# =======================================================  #


# callback to log hills and other output files
class logMeta:
    """
    Implements a Callback functor for methods.
    Logs the state of the collective variable and other parameters in metadynamics.
    """

    def __init__(self, hills_period, colvar_period, sigma, height, hillsFile, colvarFile):
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
        self.counter = 0
        self.hills_count = 0
        self.sigma = sigma
        self.height = height
        self.hillsFile = hillsFile
        self.colvarFile = colvarFile

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter != 0 and self.counter % self.hills_period == 0:

            # state.xi is the next timestep;
            # write_hills_to_file(self.counter, state.xi, self.sigma, self.height, self.hillsFile)
            write_hills_to_file(
                self.counter,
                state.xi,
                state.sigma_stride[-1],
                state.height_stride[-1],
                self.hillsFile,
            )

            self.hills_count += 1

        if self.counter % self.colvar_period == 0:

            if self.counter == 0:

                local_hills_count = 0
            else:

                local_hills_count = self.hills_count - 1

            bias_pot = 0
            bias_pot = lax.fori_loop(
                0,
                local_hills_count,
                bias_potential,
                (bias_pot, state.xi, state.xi_stride, state.sigma_stride, state.height_stride),
            )[0]

            write_colvar_to_file(self.counter, state.xi, bias_pot, self.colvarFile)

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
    """

    bias: JaxArray
    xi: JaxArray

    xi_stride: JaxArray
    sigma_stride: JaxArray
    height_stride: JaxArray

    loop: int

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class meta(SamplingMethod):
    def __init__(self, cvs, height, sigma, stride, nstrides, biasfactor, *args, **kwargs):

        super().__init__(cvs, args, kwargs)

        self.height = height
        self.sigma = sigma
        self.stride = stride
        self.nstrides = nstrides
        self.biasfactor = biasfactor

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

        self.snapshot_flags = ("positions", "indices")

    def build(self, snapshot, helpers):

        self.helpers = helpers

        return standard_meta(self, snapshot, helpers)

    # We override the default run method as Meta
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
        Direct version of the Forward Flux Sampling algorithm.

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

        context = context_generator(**context_args)
        callback = logMeta(
            self.stride,
            self.colvar_outstride,
            self.sigma,
            self.height,
            self.hillsFile,
            self.colvarFile,
        )
        wrapped_context = ContextWrapper(context, self, callback)

        with wrapped_context:

            loop = 0
            for frame in range(0, timesteps):

                sampler = wrapped_context.sampler

                run = wrapped_context.run

                # print(f"Frame: {frame}\n")

                if (frame - 1) != 0 and (frame - 1) % self.stride == 0:

                    xi_stride = sampler.state.xi_stride
                    for j in range(sampler.state.xi.shape[1]):
                        xi_stride = xi_stride.at[(loop, j)].set(sampler.state.xi[0][j])

                    loop += 1

                    sampler.state = MetaBiasState(
                        sampler.state.bias,
                        sampler.state.xi,
                        xi_stride,
                        sampler.state.sigma_stride,
                        sampler.state.height_stride,
                        loop,
                    )

                run(1, **kwargs)


def standard_meta(method, snapshot, helpers):

    cv = method.cv
    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)

    height = method.height
    sigma = method.sigma
    stride = method.stride
    nstrides = method.nstrides
    biasfactor = method.biasfactor

    colvar_outstride = method.colvar_outstride
    hillsFile = method.hillsFile
    colvarFile = method.colvarFile

    # initialize method
    def initialize():

        bias = np.zeros((natoms, 3))
        xi, _ = cv(helpers.query(snapshot))

        xi_stride = np.zeros((nstrides, xi.shape[1]))
        sigma_stride = np.full((nstrides, xi.shape[1]), sigma)
        height_stride = np.full((nstrides), height)

        loop = 0

        return MetaBiasState(bias, xi, xi_stride, sigma_stride, height_stride, loop)

    def update(state, data):

        # calculate CV and its gradient dCV/dr
        xi, Jxi = cv(data)

        bias = state.bias

        # calculate gradient bias potential along CV
        dbias_dxi = np.zeros(state.xi.shape[1])
        dbias_dxi = lax.fori_loop(
            0,
            state.loop,
            derivative_bias,
            (dbias_dxi, xi, state.xi_stride, state.sigma_stride, state.height_stride),
        )[0]

        # calculate forces
        bias = -Jxi.T @ dbias_dxi.flatten()
        bias = bias.reshape(state.bias.shape)

        return MetaBiasState(
            bias, xi, state.xi_stride, state.sigma_stride, state.height_stride, state.loop
        )

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


## HELPER FUNCTIONS
def exp_in_gaussian(delta_xi, sigma):

    sigma_square = 2 * np.multiply(sigma, sigma)
    delta_xi_square = np.multiply(delta_xi, delta_xi)
    arg = np.divide(delta_xi_square, sigma_square)

    return np.exp(-arg)


def derivative_exp_in_gaussian(delta_xi, sigma):

    arg = (delta_xi * delta_xi) / (2 * sigma * sigma)

    pre = -(delta_xi) / (sigma * sigma)

    return pre * np.exp(-arg)


def dbias_dxi_cv(j, dbias_dxi_cv_parameters):

    dbias_dxi, xi, xi_stride, sigma_cv = dbias_dxi_cv_parameters

    delta_xi = xi - xi_stride
    sigma = sigma_cv

    exp_in_gaussian_local = exp_in_gaussian(delta_xi, sigma)
    exp_in_gaussian_local = exp_in_gaussian_local.at[j].set(1)

    ncvs_product = np.product(exp_in_gaussian_local, axis=0)

    delta_xi = xi[j] - xi_stride[j]
    sigma = sigma_cv[j]

    ncvs_product *= derivative_exp_in_gaussian(delta_xi, sigma)

    dbias_dxi = dbias_dxi.at[j].set(ncvs_product)

    return dbias_dxi, xi, xi_stride, sigma_cv


def derivative_bias(i, dbias_parameters):

    dbias_dxi, xi, xi_stride, sigma_stride, height_stride = dbias_parameters

    delta_xi = xi[0] - xi_stride[i]
    height = height_stride[i]
    sigma_cv = sigma_stride[i]

    dbias_dxi_local = dbias_dxi
    dbias_dxi_local = lax.fori_loop(
        0, xi.shape[1], dbias_dxi_cv, (dbias_dxi_local, xi[0], xi_stride[i], sigma_cv)
    )[0]

    dbias_dxi = dbias_dxi + height * dbias_dxi_local

    return (dbias_dxi, xi, xi_stride, sigma_stride, height_stride)


def bias_potential(i, bias_parameters):

    bias_pot, xi, xi_stride, sigma_stride, height_stride = bias_parameters

    delta_xi = xi[0] - xi_stride[i]
    height = height_stride[i]
    sigma = sigma_stride[i]

    bias_pot = bias_pot + height * np.product(exp_in_gaussian(delta_xi, sigma), axis=0)

    return (bias_pot, xi, xi_stride, sigma_stride, height_stride)


def write_hills_to_file(frame, xi, sigma, height, hillsFile):

    with open(hillsFile, "a+") as f:

        f.write(str(frame) + "\t")
        for j in range(xi.shape[1]):

            f.write(str(xi[0][j]) + "\t")

        for j in range(sigma.shape[0]):

            f.write(str(sigma[j]) + "\t")

        f.write(str(height) + "\n")


def write_colvar_to_file(frame, xi, bias_potential, colvarFile):

    with open(colvarFile, "a+") as f:

        f.write(str(frame) + "\t")
        for j in range(xi.shape[1]):

            f.write(str(xi[0][j]) + "\t")

        f.write(str(bias_potential) + "\n")
