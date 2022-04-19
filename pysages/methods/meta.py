# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import NamedTuple, Optional

from jax import numpy as np, lax, grad, value_and_grad, vmap

from pysages.approxfun import compute_mesh
from pysages.collective_variables import get_periods, wrap
from pysages.methods.core import SamplingMethod, generalize
from pysages.utils import JaxArray, gaussian
from pysages.grids import build_indexer


class MetadynamicsState(NamedTuple):
    """
    Description of bias by metadynamics bias potential for a CV.

    Attributes
    ----------

    bias: JaxArray
        Array of metadynamics bias forces for each particle in the simulation.

    xi: JaxArray
        Collective variable value in the last simulation step.

    heights: JaxArray
        Height values for all accumulated gaussians (zeros for not yet added gaussians).

    centers: JaxArray
        Centers of the accumulated gaussians.

    sigmas: JaxArray
        Widths of the accumulated gaussians.

    bias_grad: JaxArray
        Array of metadynamics bias gradients for each particle in the simulation stored on a grid.

    bias_pot: JaxArray
        Array of metadynamics bias potentials stored on a grid.

    idx: int
        Index of the next gaussian to be deposited.

    nstep: int
        Counts the number of times `method.update` has been called.
    """

    bias: JaxArray
    xi: JaxArray
    heights: JaxArray
    centers: JaxArray
    sigmas: JaxArray
    bias_grad: JaxArray
    bias_pot: JaxArray
    idx: int
    nstep: int

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class Metadynamics(SamplingMethod):
    """
    Sampling method for metadynamics.
    """

    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, height, sigma, stride, ngaussians, deltaT=None, *args, **kwargs):
        """
        Description of method parameters.

        Arguments
        ---------

        cvs:
            Set of user selected collective variable.

        height:
            Initial height of the deposited Gaussians.

        sigma:
            Initial standard deviation of the to-be-deposit Gaussians.

        stride: int
            Bias potential deposition frequency.

        ngaussians: int
            Total number of expected gaussians (timesteps // stride + 1).

        deltaT: Optional[float] = None
            Well-tempered metadynamics $\\Delta T$ parameter
            (if `None` standard metadynamics is used).

        Keyword arguments
        -----------------

        kB: Optional[float] = 8.314462618e-3
            Boltzmann constant (must match the internal units of the backend).

        grid: Optional[Grid] = None
            If provided, the gridded version of Metadynamics will be used.
        """

        super().__init__(cvs, args, kwargs)

        self.cvs = cvs
        self.height = height
        self.sigma = sigma
        self.stride = stride
        self.ngaussians = ngaussians  # TODO: infer from timesteps and stride
        self.deltaT = deltaT

        # TODO: remove this if we eventually extract kB from the backend
        self.kB = kwargs.get("kB", 8.314462618e-3)
        self.grid = kwargs.get("grid", None)

    def build(self, snapshot, helpers):
        self.helpers = helpers
        return metadynamics(self, snapshot, helpers)


def metadynamics(method, snapshot, helpers):
    """
    Initialization and update of bias forces. Interface as expected for methods.
    """
    cv = method.cv
    height_0 = method.height
    sigma = method.sigma
    stride = method.stride
    ngaussians = method.ngaussians
    deltaT = method.deltaT
    kB = method.kB
    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)

    grid_centers = None
    if method.grid is not None:
        grid = method.grid
        dims = grid.shape.size
        grid_centers = compute_mesh(grid) * (grid.size / 2)
        get_grid_index = build_indexer(grid)

    deposit_gaussian = build_gaussian_accumulator(method)
    evaluate_bias_grad = build_bias_grad_evaluator(method)

    def initialize():
        bias = np.zeros((natoms, 3), dtype=np.float64)
        xi, _ = cv(helpers.query(snapshot))

        # TODO: for restart; use hills file to initialize corresponding arrays.
        heights = np.zeros(ngaussians, dtype=np.float64)
        centers = np.zeros((ngaussians, xi.shape[1]), dtype=np.float64)
        sigmas = np.array(sigma, dtype=np.float64)

        # Arrays to store forces and bias potential on a grid.
        if method.grid is None:
            bias_grad = bias_pot = None
        else:
            bias_grad = np.zeros((*method.grid.shape, dims), dtype=np.float64)
            bias_pot = np.zeros((*method.grid.shape,), dtype=np.float64)

        return MetadynamicsState(bias, xi, heights, centers, sigmas, bias_grad, bias_pot, 0, 0)

    def update(state, data):
        # calculate CV and the gradient of bias potential `A` along CV -- dA/dxi
        xi, Jxi = cv(data)
        I_xi = None if method.grid is None else get_grid_index(xi)
        bias = state.bias

        # deposit bias potential -- store cv centers, sigma, height depending on stride
        heights, centers, sigmas, idx, bias_grad, bias_pot = lax.cond(
            ((state.nstep + 1) % stride == 0),
            deposit_gaussian,
            lambda s: (
                s[0].heights,
                s[0].centers,
                s[0].sigmas,
                s[0].idx,
                s[0].bias_grad,
                s[0].bias_pot,
            ),
            (state, xi, I_xi, height_0, grid_centers),
        )

        # evaluate gradient of bias
        gradA = evaluate_bias_grad(xi, I_xi, heights, centers, sigmas, bias_grad)

        # calculate bias forces
        bias = -Jxi.T @ gradA.flatten()
        bias = bias.reshape(state.bias.shape)

        return MetadynamicsState(
            bias, xi, heights, centers, sigmas, bias_grad, bias_pot, idx, state.nstep + 1
        )

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


def build_gaussian_accumulator(method: Metadynamics):
    periods = get_periods(method.cvs)
    deltaT = method.deltaT
    kB = method.kB

    # Non-gridded approach
    def deposit_gaussian(params):
        state, xi, I_xi, height_0, grid_centers = params
        centers = state.centers.at[state.idx].set(xi.flatten())

        if deltaT is None:
            current_height = height_0
        else:  # if well-tempered
            V = sum_of_gaussians(xi, state.heights, centers, state.sigmas, periods)
            current_height = height_0 * np.exp(-V / (deltaT * kB))

        heights = state.heights.at[state.idx].set(current_height)

        return heights, centers, state.sigmas, state.idx + 1, state.bias_grad, state.bias_pot

    def deposit_gaussian_grid(params):
        state, xi, I_xi, height_0, grid_centers = params
        centers = state.centers.at[state.idx].set(xi.flatten())
        bias_grad = state.bias_grad
        bias_pot = state.bias_pot

        if deltaT is None:
            current_height = height_0
        else:  # if well-tempered
            current_height = height_0 * np.exp(-bias_pot[I_xi] / (deltaT * kB))

        heights = state.heights.at[state.idx].set(current_height)

        # Gaussian that is being added
        def current_gaussian(x):
            # We use sum_of_gaussians since it already takes care of the wrapping
            return sum_of_gaussians(x, current_height, xi, state.sigmas, periods)

        # Evaluate bias potential and gradient of the bias using autograd
        bias_pot_t, bias_grad_t = vmap(value_and_grad(current_gaussian))(grid_centers)
        bias_pot += bias_pot_t.reshape(bias_pot.shape)  # reshape converts to grid format
        bias_grad += bias_grad_t.reshape(bias_grad.shape)  # reshape converts to grid format

        return heights, centers, state.sigmas, state.idx + 1, bias_grad, bias_pot

    # Select method
    if method.grid is None:
        return deposit_gaussian
    else:
        return deposit_gaussian_grid


def build_bias_grad_evaluator(method: Metadynamics):
    # Select evaluation method for calculating the gradient of the bias
    if method.grid is None:
        periods = get_periods(method.cvs)

        # Non-grid-based approach
        def evaluate_bias_grad(xi, I_xi, heights, centers, sigmas, bias_grad):
            return grad(sum_of_gaussians)(xi, heights, centers, sigmas, periods)

    else:

        def evaluate_bias_grad(xi, I_xi, heights, centers, sigmas, bias_grad):
            return bias_grad[I_xi]

    return evaluate_bias_grad


# Helper function to evaluate bias potential -- may be moved to analysis part
def sum_of_gaussians(xi, heights, centers, sigmas, periods):
    """
    Sum of n-dimensional gaussians potential.
    """
    delta_x = wrap(xi - centers, periods)
    return gaussian(heights, sigmas, delta_x).sum()
