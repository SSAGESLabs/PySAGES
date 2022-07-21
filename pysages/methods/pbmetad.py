# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Implementation of Parallel Bias Well-tempered Metadynamics with optional support for grids.
"""

from typing import NamedTuple, Optional

from jax import numpy as np, grad, jit, value_and_grad, vmap
from jax.lax import cond

from pysages.approxfun import compute_mesh
from pysages.collective_variables import get_periods, wrap
from pysages.methods.core import SamplingMethod, generalize
from pysages.utils import JaxArray, identity
from pysages.grids import build_indexer
from pysages.methods.metad import MetadynamicsState, PartialMetadynamicsState


class ParallelBiasMetadynamics(SamplingMethod):
    """
    Implementation of Parallel Bias Metadynamics as described in
    [J. Chem. Theory Comput. 11, 5062–5067 (2015)](https://doi.org/10.1021/acs.jctc.5b00846)
    
    Compared to well-tempered metadynamics, the total bias potential expression and height at which
    bias is deposited for each CV is different in parallel bias metadynamics.
    """

    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, height, sigma, stride, ngaussians, *args, T, deltaT, kB, **kwargs):
        """
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

        deltaT: float = None
            Well-tempered metadynamics $\\Delta T$ parameter to set the energy
            scale for sampling.
            
        kB: float
            Boltzmann constant. It should match the internal units of the backend.

        Keyword arguments
        -----------------
        
        grid: Optional[Grid] = None
            If provided, it will be used to accelerate the computation by
            approximating the bias potential and its gradient over its centers.

        """

        super().__init__(cvs, args, kwargs)

        self.height = height
        self.sigma = sigma
        self.stride = stride
        self.ngaussians = ngaussians  # NOTE: infer from timesteps and stride
        self.T = T
        self.deltaT = deltaT
        self.kB = kwargs.get("kB", None)
        
        self.grid = kwargs.get("grid", None)

    def build(self, snapshot, helpers, *args, **kwargs):
        return _parallelbiasmetadynamics(self, snapshot, helpers)


def _parallelbiasmetadynamics(method, snapshot, helpers):
    # Initialization and update of biasing forces. Interface expected for methods.
    cv = method.cv
    stride = method.stride
    ngaussians = method.ngaussians
    natoms = np.size(snapshot.positions, 0)
    T = method.T
    deltaT = method.deltaT
    kB = method.kB
    beta = 1/(kB * T)
    kB_deltaT = kB * deltaT

    deposit_gaussian = build_gaussian_accumulator(method)
    evaluate_bias_grad = build_bias_grad_evaluator(method)

    def initialize():
        bias = np.zeros((natoms, 3), dtype=np.float64)
        xi, _ = cv(helpers.query(snapshot))

        # NOTE: for restart; use hills file to initialize corresponding arrays.
        heights = np.zeros((ngaussians, xi.size), dtype=np.float64)
        centers = np.zeros((ngaussians, xi.size), dtype=np.float64)
        sigmas = np.array(method.sigma, dtype=np.float64, ndmin=2)

        # Arrays to store forces and bias potential on a grid.
        if method.grid is None:
            grid_potential = grid_gradient = None
        else:
            shape = method.grid.shape
            grid_potential = np.zeros((*shape,), dtype=np.float64)
            grid_gradient = np.zeros((*shape, shape.size), dtype=np.float64)

        return MetadynamicsState(
            bias, xi, heights, centers, sigmas, grid_potential, grid_gradient, 0, 0
        )

    def update(state, data):
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)

        # Deposit gaussian depending on the stride
        nstep = state.nstep
        in_deposition_step = (nstep > 0) & (nstep % stride == 0)
        partial_state = deposit_gaussian(xi, state, in_deposition_step, beta, kB_deltaT)

        # Evaluate gradient of biasing potential (or generalized force)
        generalized_force = evaluate_bias_grad(partial_state)

        # Calculate biasing forces
        bias = -Jxi.T @ generalized_force.flatten()
        bias = bias.reshape(state.bias.shape)

        return MetadynamicsState(bias, *partial_state[:-1], nstep + 1)

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


def build_gaussian_accumulator(method: Metadynamics):
    """
    Returns a function that given a `MetadynamicsState`, and the value of the CV,
    stores the next gaussian that is added to the biasing potential.
    """
    periods = get_periods(method.cvs)
    height_0 = method.height
    T = method.T
    deltaT = method.deltaT
    grid = method.grid
    kB = method.kB
    beta = 1/(kB * T)

    if grid is None:
        evaluate_potential_cv = jit(lambda pstate: get_bias_each_cv(*pstate[:4], periods))
        evaluate_potential = jit(lambda pstate: parallelbias_potential(*pstate[:4], beta, periods))
    #else:
    #    evaluate_potential = jit(lambda pstate: pstate.grid_potential[pstate.grid_idx])

    def next_height(pstate, beta, kB_deltaT):
        V = evaluate_potential_cv(pstate)
        w = height_0 * np.exp(-V / (kB_deltaT)) 
        switching_probability_sum = np.sum(np.exp(-beta*V))
        switching_probability = np.exp(-beta*V)/switching_probability_sum
        return w * switching_probability

    if grid is None:
        get_grid_index = jit(lambda arg: None)
        update_grids = jit(lambda *args: (None, None))
    #else:
    #    grid_mesh = compute_mesh(grid) * (grid.size / 2)
    #    get_grid_index = build_indexer(grid)
    #    # Reshape so the dimensions are compatible
    #    accum = jit(lambda total, val: total + val.reshape(total.shape))
#
#
    #    transform = value_and_grad
    #    pack = identity
    #    update = jit(lambda V, dV, vals, grads: (accum(V, vals), accum(dV, grads)))
#
    #    def update_grids(pstate, height, xi, sigma):
    #        # We use sum_of_gaussians since it already takes care of the wrapping
    #        may be replace sum_of_gaussians with parallelbias_potential
    #        current_gaussian = jit(lambda x: sum_of_gaussians(x, height, xi, sigma, periods))
    #        # Evaluate gradient of bias (and bias potential for WT version)
    #        grid_values = pack(vmap(transform(current_gaussian))(grid_mesh))
    #        return update(pstate.grid_potential, pstate.grid_gradient, *grid_values)

    def deposit_gaussian(pstate, beta, kB_deltaT):
        xi, idx = pstate.xi, pstate.idx
        current_height = next_height(pstate, beta, kB_deltaT)
        heights = pstate.heights.at[idx].set(current_height)
        centers = pstate.centers.at[idx].set(xi.flatten())
        sigmas = pstate.sigmas
        grid_potential, grid_gradient = update_grids(pstate, current_height, xi, sigmas)
        return PartialMetadynamicsState(
            xi, heights, centers, sigmas, grid_potential, grid_gradient, idx + 1, pstate.grid_idx
        )

    def _deposit_gaussian(xi, state, in_deposition_step):
        pstate = PartialMetadynamicsState(xi, *state[2:-1], get_grid_index(xi))
        return cond(in_deposition_step, deposit_gaussian, identity, pstate)

    return _deposit_gaussian


def build_bias_grad_evaluator(method: Metadynamics):
    """
    Returns a function that given the deposited gaussians parameters, computes the
    gradient of the biasing potential with respect to the CVs.
    """
    if method.grid is None:
        periods = get_periods(method.cvs)
        T = method.T
        kB = method.kB
        beta = 1/(kB * T)
        evaluate_bias_grad = jit(lambda pstate: grad(parallelbias_potential)(*pstate[:4], beta, periods))
    #else:
    #    evaluate_bias_grad = jit(lambda pstate: pstate.grid_gradient[pstate.grid_idx])

    return evaluate_bias_grad


# Helper function to evaluate parallel bias potential
def parallelbias_potential(xi, heights, centers, sigmas, beta, periods):
    """
    Evaluate parallel bias potential Eq. 7 in 
    [J. Chem. Theory Comput. 11, 5062–5067 (2015)](https://doi.org/10.1021/acs.jctc.5b00846)
    """
    bias_each_cv = get_bias_each_cv((xi, heights, centers, sigmas, periods)
    exp_sum_gaussian = np.exp(-beta*sum_gaussian)
    
    return -(1/beta) * np.log(np.sum(exp_sum_gaussian))

# Helper function to evaluate bias potential along each CV
def get_bias_each_cv(xi, heights, centers, sigmas, periods):
    """
    Evaluate parallel bias potential along each CV.
    """
    delta_x_each_cv = wrap(xi - centers, periods)
    gaussian_each_cv = a * np.exp(-((delta_x_each_cv / sigma) ** 2) / 2)
    bias_each_cv = np.sum(gaussian_each_cv, axis=0)
    
    return bias_each_cv
                 
                 
                 