# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Implementation of Standard and Well-tempered Metadynamics
both with optional support for grids.
"""

from jax import grad, jit
from jax import numpy as np
from jax import value_and_grad, vmap
from jax.lax import cond

from pysages.approxfun import compute_mesh
from pysages.colvars import get_periods, wrap
from pysages.grids import build_indexer
from pysages.methods.core import GriddedSamplingMethod, Result, generalize
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.typing import JaxArray, NamedTuple, Optional
from pysages.utils import dispatch, gaussian, identity


class FMetadynamicsState(NamedTuple):
    """
    Metadynamics helper state

    Parameters
    ----------

    xi: JaxArray
        Collective variable value in the last simulation step.

    bias: JaxArray
        Array of Metadynamics bias forces for each particle in the simulation.

    heights: JaxArray
        Height values for all accumulated Gaussians (zeros for not yet added Gaussians).

    centers: JaxArray
        Centers of the accumulated Gaussians.

    sigmas: JaxArray
        Widths of the accumulated Gaussians.

    grid_potential: Optional[JaxArray]
        Array of Metadynamics bias potentials stored on a grid.

    grid_gradient: Optional[JaxArray]
        Array of Metadynamics bias gradients evaluated on a grid.

    idx: int
        Index of the next Gaussian to be deposited.

    ncalls: int
        Counts the number of times `method.update` has been called.

    perp: JaxArray
        Collective variable perpendicular to the Funnel_CV
    """

    xi: JaxArray
    bias: JaxArray
    heights: JaxArray
    centers: JaxArray
    sigmas: JaxArray
    grid_potential: Optional[JaxArray]
    grid_gradient: Optional[JaxArray]
    idx: int
    ncalls: int
    perp: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class PartialMetadynamicsState(NamedTuple):
    """
    Helper intermediate Metadynamics state
    """

    xi: JaxArray
    heights: JaxArray
    centers: JaxArray
    sigmas: JaxArray
    grid_potential: Optional[JaxArray]
    grid_gradient: Optional[JaxArray]
    idx: int
    grid_idx: Optional[JaxArray]


class Funnel_Metadynamics(GriddedSamplingMethod):
    """
    Implementation of Standard and Well-tempered Funnel Metadynamics as described in
    [PNAS 110.16, 6358-6363 (2013)](https://doi.org/10.1073/pnas.1303186110).
    """

    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, height, sigma, stride, ngaussians, deltaT=None, **kwargs):
        """
        Parameters
        ----------

        cvs:
            Set of user selected collective variable.

        height:
            Initial height of the deposited Gaussians.

        sigma:
            Initial standard deviation of the to-be-deposit Gaussians.

        stride: int
            Bias potential deposition frequency.

        ngaussians: int
            Total number of expected Gaussians (`timesteps // stride + 1`).

        deltaT: Optional[float] = None
            Well-tempered Metadynamics :math:`\\Delta T` parameter
            (if `None` standard Metadynamics is used).

        grid: Optional[Grid] = None
            If provided, it will be used to accelerate the computation by
            approximating the bias potential and its gradient over its centers.

        kB: Optional[float]
            Boltzmann constant. Must be provided for well-tempered Metadynamics
            simulations and should match the internal units of the backend.

        restraints: Optional[CVRestraints] = None
            If provided, it will be used to restraint CV space inside the grid.

        external_force:
            External restraint to be used for funnel calculations.
        """

        if deltaT is not None and "kB" not in kwargs:
            raise KeyError(
                "For well-tempered Metadynamics a keyword argument `kB` for "
                "the value of the Boltzmann constant (that matches the "
                "internal units of the backend) must be provided."
            )

        kwargs["grid"] = kwargs.get("grid", None)
        kwargs["restraints"] = kwargs.get("restraints", None)
        super().__init__(cvs, **kwargs)

        self.height = height
        self.sigma = sigma
        self.stride = stride
        self.ngaussians = ngaussians  # NOTE: infer from timesteps and stride
        self.deltaT = deltaT

        self.kB = kwargs.get("kB", None)

    def build(self, snapshot, helpers, *args, **kwargs):
        #        self.external_force=self.kwargs.get("external_force",lambda rs:funnel_force(rs))
        self.external_force = self.kwargs.get("external_force", None)
        return _metadynamics(self, snapshot, helpers)


def _metadynamics(method, snapshot, helpers):
    # Initialization and update of biasing forces. Interface expected for methods.
    cv = method.cv
    stride = method.stride
    ngaussians = method.ngaussians
    external_force = method.external_force
    natoms = np.size(snapshot.positions, 0)

    deposit_gaussian = build_gaussian_accumulator(method)
    evaluate_bias_grad = build_bias_grad_evaluator(method)

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        perp = 0.0
        # NOTE: for restart; use hills file to initialize corresponding arrays.
        heights = np.zeros(ngaussians, dtype=np.float64)
        centers = np.zeros((ngaussians, xi.size), dtype=np.float64)
        sigmas = np.array(method.sigma, dtype=np.float64, ndmin=2)

        # Arrays to store forces and bias potential on a grid.
        if method.grid is None:
            grid_potential = grid_gradient = None
        else:
            shape = method.grid.shape
            grid_potential = np.zeros((*shape,), dtype=np.float64) if method.deltaT else None
            grid_gradient = np.zeros((*shape, shape.size), dtype=np.float64)

        return FMetadynamicsState(
            xi, bias, heights, centers, sigmas, grid_potential, grid_gradient, 0, 0, perp
        )

    def update(state, data):
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)

        # Deposit Gaussian depending on the stride
        ncalls = state.ncalls + 1
        in_deposition_step = (ncalls > 1) & (ncalls % stride == 1)
        partial_state = deposit_gaussian(xi, state, in_deposition_step)

        # Evaluate gradient of biasing potential (or generalized force)
        generalized_force = evaluate_bias_grad(partial_state)

        # Calculate biasing forces
        bias = -Jxi.T @ generalized_force.flatten()
        eforce, perp = external_force(data)
        bias = bias.reshape(state.bias.shape) - eforce.reshape(state.bias.shape)

        return FMetadynamicsState(xi, bias, *partial_state[1:-1], ncalls, perp)

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


def build_gaussian_accumulator(method: Funnel_Metadynamics):
    """
    Returns a function that given a `FMetadynamicsState`, and the value of the CV,
    stores the next Gaussian that is added to the biasing potential.
    """
    periods = get_periods(method.cvs)
    height_0 = method.height
    deltaT = method.deltaT
    grid = method.grid
    kB = method.kB

    if deltaT is None:
        next_height = jit(lambda *args: height_0)
    else:  # if well-tempered
        if grid is None:
            evaluate_potential = jit(lambda pstate: sum_of_gaussians(*pstate[:4], periods))
        else:
            evaluate_potential = jit(lambda pstate: pstate.grid_potential[pstate.grid_idx])

        def next_height(pstate):
            V = evaluate_potential(pstate)
            return height_0 * np.exp(-V / (deltaT * kB))

    if grid is None:
        get_grid_index = jit(lambda arg: None)
        update_grids = jit(lambda *args: (None, None))
        should_deposit = jit(lambda pred, _: pred)
    else:
        grid_mesh = (compute_mesh(grid) + 1) * (grid.size / 2) + grid.lower
        get_grid_index = build_indexer(grid)
        # Reshape so the dimensions are compatible
        accum = jit(lambda total, val: total + val.reshape(total.shape))

        if deltaT is None:
            transform = grad
            pack = jit(lambda x: (x,))
            # No need to accumulate values for the potential (V is None)
            update = jit(lambda V, dV, vals: (V, accum(dV, vals)))
        else:
            transform = value_and_grad
            pack = identity
            update = jit(lambda V, dV, vals, grads: (accum(V, vals), accum(dV, grads)))

        def update_grids(pstate, height, xi, sigma):
            # We use `sum_of_gaussians` since it already takes care of the wrapping
            current_gaussian = jit(lambda x: sum_of_gaussians(x, height, xi, sigma, periods))
            # Evaluate gradient of bias (and bias potential for WT version)
            grid_values = pack(vmap(transform(current_gaussian))(grid_mesh))
            return update(pstate.grid_potential, pstate.grid_gradient, *grid_values)

        def should_deposit(in_deposition_step, I_xi):
            in_bounds = ~(np.any(np.array(I_xi) == grid.shape))
            return in_deposition_step & in_bounds

    def deposit_gaussian(pstate):
        xi, idx = pstate.xi, pstate.idx
        current_height = next_height(pstate)
        heights = pstate.heights.at[idx].set(current_height)
        centers = pstate.centers.at[idx].set(xi.flatten())
        sigmas = pstate.sigmas
        grid_potential, grid_gradient = update_grids(pstate, current_height, xi, sigmas)
        return PartialMetadynamicsState(
            xi, heights, centers, sigmas, grid_potential, grid_gradient, idx + 1, pstate.grid_idx
        )

    def _deposit_gaussian(xi, state, in_deposition_step):
        I_xi = get_grid_index(xi)
        pstate = PartialMetadynamicsState(xi, *state[2:-2], I_xi)
        predicate = should_deposit(in_deposition_step, I_xi)
        return cond(predicate, deposit_gaussian, identity, pstate)

    return _deposit_gaussian


def build_bias_grad_evaluator(method: Funnel_Metadynamics):
    """
    Returns a function that given the deposited Gaussians parameters, computes the
    gradient of the biasing potential with respect to the CVs.
    """
    grid = method.grid
    restraints = method.restraints
    if grid is None:
        periods = get_periods(method.cvs)
        evaluate_bias_grad = jit(lambda pstate: grad(sum_of_gaussians)(*pstate[:4], periods))
    else:
        if restraints:

            def ob_force(pstate):  # out-of-bounds force
                lo, hi, kl, kh = restraints
                xi, *_ = pstate
                xi = pstate.xi.reshape(grid.shape.size)
                force = apply_restraints(lo, hi, kl, kh, xi)
                return force

        else:

            def ob_force(pstate):  # out-of-bounds force
                return np.zeros(grid.shape.size)

        def get_force(pstate):
            return pstate.grid_gradient[pstate.grid_idx]

        def evaluate_bias_grad(pstate):
            ob = np.any(np.array(pstate.grid_idx) == grid.shape)  # out of bounds
            return cond(ob, ob_force, get_force, pstate)

    return evaluate_bias_grad


# Helper function to evaluate bias potential -- may be moved to analysis part
def sum_of_gaussians(xi, heights, centers, sigmas, periods):
    """
    Sum of n-dimensional Gaussians potential.
    """
    delta_x = wrap(xi - centers, periods)
    return gaussian(heights, sigmas, delta_x).sum()


@dispatch
def analyze(result: Result[Funnel_Metadynamics]):
    """
    Helper for calculating the free energy from the final state of a `Metadynamics` run.

    Parameters
    ----------

    result: Result[Metadynamics]:
        Result bundle containing method, final metadynamics state, and callback.

    Returns
    -------

    dict:
        A dictionary with the following keys:

        heights: JaxArray
            Height of the Gaussian bias potential during the simulation.

        metapotential: Callable
            Maps a user-provided array of CV values to the corresponding deposited bias
            potential. For standard metadynamics, the free energy along user-provided CV
            range is the same as `metapotential(cv)`.
            In the case of well-tempered metadynamics, the free energy is equal to
            `(T + deltaT) / deltaT * metapotential(cv)`, where `T` is the simulation
            temperature and `deltaT` is the user-defined parameter in
            well-tempered metadynamics.
    """
    method = result.method
    states = result.states

    P = get_periods(method.cvs)

    if len(states) == 1:
        heights = states[0].heights
        centers = states[0].centers
        sigmas = states[0].sigmas

        metapotential = jit(vmap(lambda x: sum_of_gaussians(x, heights, centers, sigmas, P)))

        return dict(heights=heights, metapotential=metapotential)

    # For multiple-replicas runs we return a list heights and functions
    # (one for each replica)

    def build_metapotential(heights, centers, sigmas):
        return jit(vmap(lambda x: sum_of_gaussians(x, heights, centers, sigmas, P)))

    heights = []
    metapotentials = []

    for s in states:
        heights.append(s.heights)
        metapotentials.append(build_metapotential(s.heights, s.centers, s.sigmas))

    ana_result = dict(heights=heights, metapotential=metapotentials)
    return numpyfy_vals(ana_result)
