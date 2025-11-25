# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Implementation of Standard and Well-tempered MetaD-ABF
both with support for grids.
"""

from jax import grad, jit
from jax import numpy as np
from jax import value_and_grad, vmap
from jax.lax import cond

from pysages.approxfun import compute_mesh
from pysages.colvars import get_periods, wrap
from pysages.grids import build_indexer
from pysages.methods.analysis import GradientLearning, _funnelanalyze
from pysages.methods.core import GriddedSamplingMethod, Result, generalize
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.typing import JaxArray, NamedTuple, Optional
from pysages.utils import dispatch, gaussian, identity, linear_solver


class FMetaDABFState(NamedTuple):
    """
    MetaDABF helper state

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

    idg: int
        Index of the CV in the forces array.

    hist: JaxArray (grid.shape)
        Histogram of visits to the bins in the collective variable grid.

    Fsum: JaxArray (grid.shape, CV shape)
        Cumulative forces at each bin in the CV grid.

    force: JaxArray (grid.shape, CV shape)
        Average force at each bin of the CV grid.

    Wp: JaxArray (CV shape)
        Product of W matrix and momenta matrix for the current step.

    Wp_: JaxArray (CV shape)
        Product of W matrix and momenta matrix for the previous step.

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
    idg: int
    Wp: JaxArray
    Wp_: JaxArray
    force: JaxArray
    Fsum: JaxArray
    Frestr: JaxArray
    hist: JaxArray
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


class Funnel_MetaDABF(GriddedSamplingMethod):
    """
    Implementation of Standard and Well-tempered Funnel MetaDABF as described in
    arXiv preprint arXiv:2504.13575.
    """

    snapshot_flags = {"positions", "indices", "momenta"}

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
        self.use_pinv = self.kwargs.get("use_pinv", False)

    def build(self, snapshot, helpers, *args, **kwargs):
        self.external_force = self.kwargs.get("external_force", None)
        return _metadynamics(self, snapshot, helpers)


def _metadynamics(method, snapshot, helpers):
    # Initialization and update of biasing forces. Interface expected for methods.
    cv = method.cv
    stride = method.stride
    dims = method.grid.shape.size
    dt = snapshot.dt
    ngaussians = method.ngaussians
    get_grid_index = build_indexer(method.grid)
    external_force = method.external_force
    tsolve = linear_solver(method.use_pinv)
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
        hist = np.zeros(method.grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*method.grid.shape, dims))
        force = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        restr = np.zeros(dims)
        Frestr = np.zeros((*method.grid.shape, dims))
        # Arrays to store forces and bias potential on a grid.
        if method.grid is None:
            grid_potential = grid_gradient = None
        else:
            shape = method.grid.shape
            grid_potential = np.zeros((*shape,), dtype=np.float64) if method.deltaT else None
            grid_gradient = np.zeros((*shape, shape.size), dtype=np.float64)

        return FMetaDABFState(
            xi,
            bias,
            heights,
            centers,
            sigmas,
            grid_potential,
            grid_gradient,
            0,
            0,
            Wp,
            Wp_,
            force,
            Fsum,
            Frestr,
            hist,
            0,
            perp,
        )

    def update(state, data):
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)

        # Deposit Gaussian depending on the stride
        ncalls = state.ncalls + 1
        in_deposition_step = (ncalls > 1) & (ncalls % stride == 1)
        partial_state = deposit_gaussian(xi, state, in_deposition_step)

        # Evaluate gradient of biasing potential (or generalized force)
        force = evaluate_bias_grad(partial_state)

        # Calculate biasing forces
        bias = -Jxi.T @ force.flatten()
        eforce, perp = external_force(data)
        bias = bias.reshape(state.bias.shape) - eforce.reshape(state.bias.shape)
        p = data.momenta
        Wp = tsolve(Jxi, p)
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_xi = get_grid_index(xi)
        hist = state.hist.at[I_xi].add(1)
        Fsum = state.Fsum.at[I_xi].add(dWp_dt + state.force)
        restr = tsolve(Jxi, eforce.reshape(p.shape))
        #
        Frestr = state.Frestr.at[I_xi].add(restr)

        return FMetaDABFState(
            xi,
            bias,
            *partial_state[1:-1],
            I_xi,
            Wp,
            state.Wp,
            force,
            Fsum,
            Frestr,
            hist,
            ncalls,
            perp
        )

    return snapshot, initialize, generalize(update, helpers, jit_compile=True)


def build_gaussian_accumulator(method: Funnel_MetaDABF):
    """
    Returns a function that given a `FMetaDABFState`, and the value of the CV,
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
        grid_mesh = compute_mesh(grid)
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
        pstate = PartialMetadynamicsState(xi, *state[2:-9], I_xi)
        predicate = should_deposit(in_deposition_step, I_xi)
        return cond(predicate, deposit_gaussian, identity, pstate)

    return _deposit_gaussian


def build_bias_grad_evaluator(method: Funnel_MetaDABF):
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
def analyze(result: Result[Funnel_MetaDABF], **kwargs):
    """
    Computes the free energy from the result of an `Funnel_MetaDABF` run.
    Integrates the forces via a gradient learning strategy.

    Parameters
    ----------

    result: Result[Funnel_ABF]:
        Result bundle containing method, final ABF state, and callback.

    topology: Optional[Tuple[int]] = (8, 8)
        Defines the architecture of the neural network
        (number of nodes in each hidden layer).

    Returns
    -------

    dict:
        A dictionary with the following keys:

        histogram: JaxArray
            Histogram for the states visited during the method.

        mean_force: JaxArray
            Average force at each bin of the CV grid.

        free_energy: JaxArray
            Free Energy at each bin of the CV grid.

        mesh: JaxArray
            Grid used in the method.

        fes_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy in the
            CV domain defined by the grid.

        corrected_force: JaxArray
            Average mean force without restraint at each bin of the CV grid.

        corrected_energy: JaxArray
            Free Energy without restraint at each bin of the CV grid.

        corr_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy without restraints
            in the CV domain defined by the grid.

        restraint_force: JaxArray
            Average mean force of the restraints at each bin of the CV grid.

        restraint_energy: JaxArray
            Free Energy of the restraints at each bin of the CV grid.

        restr_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy of the restraints
            in the CV domain defined by the grid.



    NOTE:
    For multiple-replicas runs we return a list (one item per-replica)
    for each attribute.
    """
    topology = kwargs.get("topology", (8, 8))
    _result = _funnelanalyze(result, GradientLearning(), topology)
    return numpyfy_vals(_result)
