# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Spectral Adaptive Biasing Force Sampling Method.

SpectralABF learns the generalized mean forces and free energy as functions of some
collective variables by fitting a basis functions expansion from a binned estimate of the
mean forces.

It closely follows the Adaptive Biasing Force (ABF) algorithm, except for biasing the
simulation, which is done from the continuous approximation to the generalized mean force
provided by the basis functions expansion.
"""

from jax import jit
from jax import numpy as np
from jax.lax import cond

from pysages.approxfun import (
    Fun,
    SpectralGradientFit,
    build_evaluator,
    build_fitter,
    build_grad_evaluator,
    compute_mesh,
)
from pysages.grids import Chebyshev, Grid, build_indexer, convert, grid_transposer
from pysages.methods.core import GriddedSamplingMethod, Result, generalize
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.typing import JaxArray, NamedTuple, Tuple
from pysages.utils import dispatch, first_or_all, linear_solver


class SpectralABFState(NamedTuple):
    """
    SpectralABF internal state.

    Parameters
    ----------

    xi: JaxArray [cvs.shape]
        Last collective variable recorded in the simulation.

    bias: JaxArray [natoms, 3]
        Array with biasing forces for each particle.

    hist: JaxArray [grid.shape]
        Histogram of visits to the bins in the collective variable grid.

    Fsum: JaxArray [grid.shape, cvs.shape]
        The cumulative force recorded at each bin of the CV grid.

    force: JaxArray [grid.shape, cvs.shape]
        Average force at each bin of the CV grid.

    Wp: JaxArray [cvs.shape]
        Estimate of the product $W p$ where `p` is the matrix of momenta and
        `W` the Moore-Penrose inverse of the Jacobian of the CVs.

    Wp_: JaxArray [cvs.shape]
        The value of `Wp` for the previous integration step.

    fun: Fun
        Object that holds the coefficients of the basis functions
        approximation to the free energy.

    ncalls: int
        Counts the number of times the method's update has been called.
    """

    xi: JaxArray
    bias: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    force: JaxArray
    Wp: JaxArray
    Wp_: JaxArray
    fun: Fun
    ncalls: int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialSpectralABFState(NamedTuple):
    xi: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    ind: Tuple
    fun: Fun
    pred: bool


class SpectralABF(GriddedSamplingMethod):
    """
    Implementation of the Spectral ABF method described in
    "SIRENs and Sobolev Sampling of Free Energy Landscapes"
    (https://arxiv.org/abs/2202.01876).

    Parameters
    ----------

    cvs: Union[List, Tuple]
        Set of user selected collective variable.

    grid: Grid
        Specifies the collective variables domain and number of bins for discretizing
        the CV space along each CV dimension. For non-periodic grids this will be
        converted to a Chebyshev-distributed grid.

    N: Optional[int] = 500
        Threshold parameter before accounting for the full average
        of the adaptive biasing force.

    fit_freq: Optional[int] = 100
        Fitting frequency.

    fit_threshold: Optional[int] = 500
        Number of time steps after which fitting starts to take place.

    restraints: Optional[CVRestraints] = None
        If provided, indicate that harmonic restraints will be applied when any
        collective variable lies outside the box from `restraints.lower` to
        `restraints.upper`.

    use_pinv: Optional[Bool] = False
        If set to True, the product `W @ p` will be estimated using
        `np.linalg.pinv` rather than using the `scipy.linalg.solve` function.
        This is computationally more expensive but numerically more stable.
    """

    snapshot_flags = {"positions", "indices", "momenta"}

    def __init__(self, cvs, grid, **kwargs):
        super().__init__(cvs, grid, **kwargs)
        self.N = np.asarray(self.kwargs.get("N", 500))
        self.fit_freq = self.kwargs.get("fit_freq", 100)
        self.fit_threshold = self.kwargs.get("fit_threshold", 500)
        self.grid = self.grid if self.grid.is_periodic else convert(self.grid, Grid[Chebyshev])
        self.model = SpectralGradientFit(self.grid)
        self.use_pinv = self.kwargs.get("use_pinv", False)

    def build(self, snapshot, helpers, *_args, **_kwargs):
        """
        Returns the `initialize` and `update` functions for the sampling method.
        """
        return _spectral_abf(self, snapshot, helpers)


def _spectral_abf(method, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    fit_freq = method.fit_freq
    fit_threshold = method.fit_threshold

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)

    # Helper methods
    tsolve = linear_solver(method.use_pinv)
    get_grid_index = build_indexer(grid)
    fit = build_fitter(method.model)
    fit_forces = build_free_energy_fitter(method, fit)
    estimate_force = build_force_estimator(method)

    query, dimensionality, to_force_units = helpers

    def initialize():
        xi, _ = cv(query(snapshot))
        bias = np.zeros((natoms, dimensionality()))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        force = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        fun = fit(Fsum)
        return SpectralABFState(xi, bias, hist, Fsum, force, Wp, Wp_, fun, 0)

    def update(state, data):
        # During the intial stage use ABF
        ncalls = state.ncalls + 1
        in_fitting_regime = ncalls > fit_threshold
        in_fitting_step = in_fitting_regime & (ncalls % fit_freq == 1)
        # Fit forces
        fun = fit_forces(state, in_fitting_step)
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        #
        p = data.momenta
        Wp = tsolve(Jxi, p)
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_xi = get_grid_index(xi)
        hist = state.hist.at[I_xi].add(1)
        Fsum = state.Fsum.at[I_xi].add(to_force_units(dWp_dt) + state.force)
        #
        force = estimate_force(
            PartialSpectralABFState(xi, hist, Fsum, I_xi, fun, in_fitting_regime)
        )
        bias = np.reshape(-Jxi.T @ force, state.bias.shape)
        #
        return SpectralABFState(xi, bias, hist, Fsum, force, Wp, state.Wp, fun, ncalls)

    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_fitter(_method: SpectralABF, fit):
    """
    Returns a function that given a `SpectralABFState` performs a least squares fit of the
    generalized average forces for finding the coefficients of a basis functions expansion
    of the free energy.
    """

    def _fit_forces(state):
        shape = (*state.Fsum.shape[:-1], 1)
        force = state.Fsum / np.maximum(state.hist.reshape(shape), 1)
        return fit(force)

    def skip_fitting(state):
        return state.fun

    def fit_forces(state, in_fitting_step):
        return cond(in_fitting_step, _fit_forces, skip_fitting, state)

    return fit_forces


@dispatch
def build_force_estimator(method: SpectralABF):
    """
    Returns a function that given the coefficients of basis functions expansion and a CV
    value, evaluates the function approximation to the gradient of the free energy.
    """
    N = method.N
    grid = method.grid
    dims = grid.shape.size
    model = method.model
    get_grad = build_grad_evaluator(model)

    def average_force(state):
        i = state.ind
        return state.Fsum[i] / np.maximum(state.hist[i], N)

    def interpolate_force(state):
        return get_grad(state.fun, state.xi).reshape(grid.shape.size)

    def _estimate_force(state):
        return cond(state.pred, interpolate_force, average_force, state)

    if method.restraints is None:
        ob_force = jit(lambda state: np.zeros(dims))
    else:
        lo, hi, kl, kh = method.restraints

        def ob_force(state):
            xi = state.xi.reshape(grid.shape.size)
            return apply_restraints(lo, hi, kl, kh, xi)

    def estimate_force(state):
        ob = np.any(np.array(state.ind) == grid.shape)  # Out of bounds condition
        return cond(ob, ob_force, _estimate_force, state)

    return estimate_force


@dispatch
def analyze(result: Result[SpectralABF]):
    """
    Parameters
    ----------

    result: Result[SpectralABF]
        Result bundle containing the method, final states, and callbacks.

    dict:
        A dictionary with the following keys:

        histogram: JaxArray
            A histogram of the visits to each bin in the CV grid.

        mean_force: JaxArray
            Generalized mean forces at each bin in the CV grid.

        free_energy: JaxArray
            Free energy at each bin in the CV grid.

        mesh: JaxArray
            These are the values of the CVs that are used as inputs for training.

        fun: Fun
            Coefficients of the basis functions expansion approximating the free energy.

        fes_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy in the CV domain defined
            by the grid.

    NOTE:
    For multiple-replicas runs we return a list (one item per-replica) for each attribute.
    """
    method = result.method

    grid = method.grid
    mesh = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    evaluate = build_evaluator(method.model)

    def average_forces(hist, Fsum):
        hist = np.expand_dims(hist, hist.ndim)
        return Fsum / np.maximum(hist, 1)

    def build_fes_fn(fun):
        def fes_fn(x):
            A = evaluate(fun, x)
            return A.max() - A

        return jit(fes_fn)

    hists = []
    mean_forces = []
    free_energies = []
    funs = []
    fes_fns = []

    # We transpose the data for convenience when plotting
    transpose = grid_transposer(grid)
    d = mesh.shape[-1]

    for s in result.states:
        fes_fn = build_fes_fn(s.fun)
        hists.append(transpose(s.hist))
        mean_forces.append(transpose(average_forces(s.hist, s.Fsum)))
        free_energies.append(transpose(fes_fn(mesh)))
        funs.append(s.fun)
        fes_fns.append(fes_fn)

    ana_result = {
        "histogram": first_or_all(hists),
        "mean_force": first_or_all(mean_forces),
        "free_energy": first_or_all(free_energies),
        "mesh": transpose(mesh).reshape(-1, d).squeeze(),
        "fun": first_or_all(funs),
        "fes_fn": first_or_all(fes_fns),
    }

    return numpyfy_vals(ana_result)
