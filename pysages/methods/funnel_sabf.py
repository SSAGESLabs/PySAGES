# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Funnel Spectral Adaptive Biasing Force Sampling Method.

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


class SpectralFABFState(NamedTuple):
    """
    Funnel_SpectralABF internal state.

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

    restr: JaxArray [grid.shape, cvs.shape]
        Instantaneous restraint force at each bin of the CV grid.

    proj: JaxArray [cvs.shape]
        Last collective variable from restraints recorded in the simulation.

    Frestr: JaxArray [grid.shape, cvs.shape]
        The cumulative restraint force recorded at each bin of the CV grid.

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
    restr: JaxArray
    proj: JaxArray
    Frestr: JaxArray
    ncalls: int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialSpectralFABFState(NamedTuple):
    xi: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    ind: Tuple
    fun: Fun
    pred: bool


class Funnel_SpectralABF(GriddedSamplingMethod):
    """
    Implementation of the Funnel_Spectral ABF method described in

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

    ext_force: Optional[ext_force] = None
        If provided, indicate the geometric restraints will be applied for any
        collective variables.
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
        self.ext_force = self.kwargs.get("ext_force", None)
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
    ext_force = method.ext_force

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        force = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        fun = fit(Fsum)
        restr = np.zeros(dims)
        proj = 0.0
        Frestr = np.zeros((*grid.shape, dims))
        return SpectralFABFState(xi, bias, hist, Fsum, force, Wp, Wp_, fun, restr, proj, Frestr, 0)

    def update(state, data):
        # During the intial stage use ABF
        ncalls = state.ncalls + 1
        in_fitting_regime = ncalls > fit_threshold
        in_fitting_step = in_fitting_regime & (ncalls % fit_freq == 1)
        # Fit forces
        fun = fit_forces(state, in_fitting_step)
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        # Restraint force and logger
        e_f, proj = ext_force(data)
        #
        p = data.momenta
        Wp = tsolve(Jxi, p)
        # Second order backward finite difference
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_xi = get_grid_index(xi)
        hist = state.hist.at[I_xi].add(1)
        Fsum = state.Fsum.at[I_xi].add(dWp_dt + state.force)
        #
        force = estimate_force(
            PartialSpectralFABFState(xi, hist, Fsum, I_xi, fun, in_fitting_regime)
        )
        bias = np.reshape(-Jxi.T @ force, state.bias.shape) + np.reshape(-e_f, state.bias.shape)
        # Restraint contribution to force
        restr = tsolve(Jxi, e_f.reshape(p.shape))
        Frestr = state.Frestr.at[I_xi].add(restr)
        return SpectralFABFState(
            xi, bias, hist, Fsum, force, Wp, state.Wp, fun, restr, proj, Frestr, ncalls
        )

    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_fitter(_method: Funnel_SpectralABF, fit):
    """
    Returns a function that given a `SpectralFABFState` performs a least squares fit of the
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
def build_force_estimator(method: Funnel_SpectralABF):
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
def analyze(result: Result[Funnel_SpectralABF]):
    """
    Parameters
    ----------

    result: Result[FunnelSpectralABF]
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

        fun: Fun
            Coefficients of the basis functions expansion approximating the free energy.

        fes_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy in the CV domain defined
            by the grid.

        fun_corr: Fun
            Coefficients of the basis functions expansion approximating the free energy
            defined without external restraints.

        corr_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy without external restraints
            in the CV domain defined by the grid.
        fun_rstr: Fun
            Coefficients of the basis functions expansion approximating the free energy
            of the external restraints.

        rstr_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy of theexternal restraints
            in the CV domain defined by the grid.

        corrected_force: JaxArray
            Generalized mean forces without restraints at each bin in the CV grid.

        corrected_energy: JaxArray
            Free energy without restraints at each bin in the CV grid.

        restraint_force: JaxArray
            Generalized mean forces of the restraints at each bin in the CV grid.

        restraint_energy: JaxArray
            Free energy of restraints at each bin in the CV grid.


    NOTE:
    For multiple-replicas runs we return a list (one item per-replica) for each attribute.
    """
    method = result.method
    fit = build_fitter(method.model)
    grid = method.grid
    mesh = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    evaluate = build_evaluator(method.model)

    def average_forces(hist, Fsum):
        hist = np.expand_dims(hist, hist.ndim)
        return Fsum / np.maximum(hist, 1)

    def fit_corr(state):
        shape = (*state.Fsum.shape[:-1], 1)
        force = (state.Fsum + state.Frestr) / np.maximum(state.hist.reshape(shape), 1)
        return fit(force)

    def fit_restr(state):
        shape = (*state.Fsum.shape[:-1], 1)
        force = (state.Frestr) / np.maximum(state.hist.reshape(shape), 1)
        return fit(force)

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
    forces_corrected = []
    corrected_energies = []
    funs_corr = []
    fes_corr = []
    restraint_forces = []
    restraint_energies = []
    funs_rstr = []
    fes_rstr = []
    # We transpose the data for convenience when plotting
    transpose = grid_transposer(grid)
    d = mesh.shape[-1]

    for s in result.states:
        fun_corr = fit_corr(s)
        fun_rstr = fit_restr(s)
        fes_fn = build_fes_fn(s.fun)
        corr_fn = build_fes_fn(fun_corr)
        restr_fn = build_fes_fn(fun_rstr)
        hists.append(transpose(s.hist))
        mean_forces.append(transpose(average_forces(s.hist, s.Fsum)))
        free_energies.append(transpose(fes_fn(mesh)))
        funs.append(s.fun)
        fes_fns.append(fes_fn)
        forces_corrected.append(transpose(average_forces(s.hist, s.Fsum + s.Frestr)))
        corrected_energies.append(transpose(corr_fn(mesh)))
        funs_corr.append(fun_corr)
        fes_corr.append(corr_fn)
        restraint_forces.append(transpose(average_forces(s.hist, s.Frestr)))
        restraint_energies.append(transpose(restr_fn(mesh)))
        funs_rstr.append(fun_rstr)
        fes_rstr.append(restr_fn)

    ana_result = {
        "histogram": first_or_all(hists),
        "mean_force": first_or_all(mean_forces),
        "free_energy": first_or_all(free_energies),
        "mesh": transpose(mesh).reshape(-1, d).squeeze(),
        "fun": first_or_all(funs),
        "fes_fn": first_or_all(fes_fns),
        "corrected_force": first_or_all(forces_corrected),
        "corrected_energy": first_or_all(corrected_energies),
        "fun_corr": first_or_all(funs_corr),
        "corr_fn": first_or_all(fes_corr),
        "restraint_force": first_or_all(restraint_forces),
        "restraint_energy": first_or_all(restraint_energies),
        "fun_rstr": first_or_all(funs_rstr),
        "rstr_fn": first_or_all(fes_rstr),
    }

    return numpyfy_vals(ana_result)
