# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
SIRENs Sampling

Sirens is a free energy sampling method which uses artificial neural networks to
generate an on-the-fly adaptive bias capable of rapidly resolving free energy landscapes.
The method learns both from the frequency of visits to bins in a CV-space and generalized
force estimates. It can be seen as a generalization of ANN and FUNN sampling methods that
uses one neural network to approximate the free energy and its derivatives.
"""

import numbers
from functools import partial

from jax import grad, jit
from jax import numpy as np
from jax import vmap
from jax.lax import cond

from pysages.approxfun import compute_mesh
from pysages.approxfun import scale as _scale
from pysages.grids import build_indexer, grid_transposer
from pysages.methods.core import NNSamplingMethod, Result, generalize
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.ml import objectives
from pysages.ml.models import Siren
from pysages.ml.objectives import GradientsSSE, L2Regularization, Sobolev1SSE
from pysages.ml.optimizers import LevenbergMarquardt
from pysages.ml.training import NNData, build_fitting_function, convolve
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.typing import JaxArray, NamedTuple, Tuple
from pysages.utils import dispatch, first_or_all, linear_solver


class SirensState(NamedTuple):  # pylint: disable=R0903
    """
    Parameters
    ----------

    xi: JaxArray
        Last collective variable recorded in the simulation.

    bias: JaxArray
        Array with biasing forces for each particle.

    hist: JaxArray
        Histogram of visits to the bins in the collective variable grid.

    histp: JaxArray
        Partial histogram of visits to the bins in the collective variable grid,
        resets to zero after each training sweep.

    prob: JaxArray
        Probability distribution of the CV evaluated ath each bin of the CV grid.

    fe: JaxArray
        Current estimate of the free energy evaluated at each bin of the CV grid.

    Fsum: JaxArray
        The cumulative force recorded at each bin of the CV grid.

    force: JaxArray
        Average force at each bin of the CV grid.

    Wp: JaxArray
        Estimate of the product $W p$ where `p` is the matrix of momenta and
        `W` the Moore-Penrose inverse of the Jacobian of the CVs.

    Wp_: JaxArray
        The value of `Wp` for the previous integration step.

    nn: NNDada
        Bundle of the neural network parameters, and output scaling coefficients.

    ncalls: int
        Counts the number of times the method's update has been called.
    """

    xi: JaxArray
    bias: JaxArray
    hist: JaxArray
    histp: JaxArray
    prob: JaxArray
    fe: JaxArray
    Fsum: JaxArray
    force: JaxArray
    Wp: JaxArray
    Wp_: JaxArray
    nn: NNData
    ncalls: int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialSirensState(NamedTuple):  # pylint: disable=C0115,R0903
    xi: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    ind: Tuple
    nn: NNData
    pred: bool


class Sirens(NNSamplingMethod):
    """
    Implementation of the sampling method described in
    "Sobolev Sampling of Free Energy Landscapes"

    Parameters
    ----------
    cvs: Union[List, Tuple]
        List of collective variables.

    grid: Grid
        Specifies the CV domain and number of bins for discretizing the CV space
        along each CV dimension.

    topology: Tuple[int]
        Defines the architecture of the neural network
        (number of nodes of each hidden layer).

    mode: Literal["abf", "cff"]
        If `mode == "cff"`, the model will be trained in both histogram of visits and mean
        forces information. Otherwise, only forces will be used as in `ABF` and `FUNN`.

    kT: Optional[numbers.Real]
        Value of `kT` in the same units as the backend internal energy units.
        When `mode == "cff"` this parameter has to be a machine real, otherwise it is
        ignored even if provided.

    N: Optional[int] = 500
        Threshold parameter before accounting for the full average of the
        binned generalized mean force.

    train_freq: Optional[int] = 5000
        Training frequency.

    optimizer: Optional[Optimizer] = None
        Optimization method used for training. Must be compatible with the selected `mode`.

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

    def __init__(self, cvs, grid, topology, **kwargs):
        mode = kwargs.get("mode", "abf")
        kT = kwargs.get("kT", None)

        regularizer = L2Regularization(1e-4)
        loss = GradientsSSE() if mode == "abf" else Sobolev1SSE()
        default_optimizer = LevenbergMarquardt(loss=loss, reg=regularizer, max_iters=250)
        optimizer = kwargs.get("optimizer", default_optimizer)

        self.__check_init_invariants__(mode, kT, optimizer)

        super().__init__(cvs, grid, topology, **kwargs)

        self.mode = mode
        self.kT = kT
        self.N = np.asarray(self.kwargs.get("N", 500))
        self.train_freq = self.kwargs.get("train_freq", 5000)

        dims = grid.shape.size
        scale = partial(_scale, grid=grid)
        self.model = Siren(dims, 1, topology, transform=scale)
        self.optimizer = optimizer
        self.use_pinv = self.kwargs.get("use_pinv", False)

    def __check_init_invariants__(self, mode, kT, optimizer):
        if mode not in ("abf", "cff"):
            raise ValueError(f"Invalid mode {mode}. Possible values are 'abf' and 'cff'")

        if mode == "cff":
            if kT is None:
                raise KeyError(
                    "When running in 'cff' mode, a keyword argument `kT` in the same "
                    "units as the backend internal energy units must be provided."
                )
            assert isinstance(kT, numbers.Real)
            assert isinstance(optimizer.loss, objectives.Sobolev1Loss)
        else:
            assert isinstance(optimizer.loss, objectives.GradientsLoss)

    def build(self, snapshot, helpers, *_args, **_kwargs):
        return _sirens(self, snapshot, helpers)


def _sirens(method: Sirens, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    train_freq = method.train_freq
    dt = snapshot.dt

    # Neural network paramters
    ps, _ = unpack(method.model.parameters)

    # Helper methods
    tsolve = linear_solver(method.use_pinv)
    get_grid_index = build_indexer(grid)
    learn_free_energy = build_free_energy_learner(method)
    estimate_force = build_force_estimator(method)

    query, dimensionality, to_force_units = helpers

    if method.mode == "cff":
        increase_me_maybe = jit(lambda a, idx, v: a.at[idx].add(v))
    else:
        increase_me_maybe = jit(lambda a, idx, v: a)

    def initialize():
        dims = grid.shape.size
        natoms = np.size(snapshot.positions, 0)
        gshape = grid.shape if dims > 1 else (*grid.shape, 1)

        xi, _ = cv(query(snapshot))
        bias = np.zeros((natoms, dimensionality()))
        hist = np.zeros(gshape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        force = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        nn = NNData(ps, np.array(0.0), np.array(1.0))

        if method.mode == "cff":
            histp = np.zeros(gshape, dtype=np.uint32)
            prob = np.zeros(gshape)
            fe = np.zeros(gshape)
        else:
            histp = prob = fe = None

        return SirensState(xi, bias, hist, histp, prob, fe, Fsum, force, Wp, Wp_, nn, 0)

    def update(state, data):
        # During the intial stage, when there are not enough collected samples, use ABF
        ncalls = state.ncalls + 1
        in_training_regime = ncalls > train_freq
        in_training_step = in_training_regime & (ncalls % train_freq == 1)
        histp, prob, fe, nn = learn_free_energy(state, in_training_step)
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        #
        p = data.momenta
        Wp = tsolve(Jxi, p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_xi = get_grid_index(xi)
        hist = state.hist.at[I_xi].add(1)
        Fsum = state.Fsum.at[I_xi].add(to_force_units(dWp_dt) + state.force)
        histp = increase_me_maybe(histp, I_xi, 1)  # Special handling depending on the mode
        #
        force = estimate_force(PartialSirensState(xi, hist, Fsum, I_xi, nn, in_training_regime))
        bias = (-Jxi.T @ force).reshape(state.bias.shape)
        #
        return SirensState(xi, bias, hist, histp, prob, fe, Fsum, force, Wp, state.Wp, nn, ncalls)

    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_learner(method: Sirens):
    """
    Returns a function that given a `SirensState` trains the method's neural networks which
    give estimates of the probability density and the gradient of the free energy.
    """
    # Local aliases
    f32 = np.float32

    mode = method.mode
    kT = method.kT
    grid = method.grid
    model = method.model
    optimizer = method.optimizer

    dims = grid.shape.size
    shape = (*grid.shape, 1)
    inputs = f32((compute_mesh(grid) + 1) * grid.size / 2 + grid.lower)
    smoothing_kernel = f32(blackman_kernel(dims, 5))
    padding = "wrap" if grid.is_periodic else "edge"
    conv = partial(convolve, kernel=smoothing_kernel, boundary=padding)

    _, layout = unpack(model.parameters)
    fit = build_fitting_function(model, optimizer)
    normalize_gradients = _build_gradient_normalizer(grid)

    def vsmooth(y):
        return vmap(conv)(y.T).T

    smooth = conv if dims > 1 else vsmooth

    if mode == "cff":

        def preprocess(y, dy):
            dy, dy_std = normalize_gradients(dy)
            s = np.maximum(y.std(), dy_std)
            return (smooth(f32(y / s)), dy / s), s

        def preupdate_free_energy(state):
            prob = state.prob + state.histp * np.exp(state.fe / kT)
            fe = kT * np.log(np.maximum(1, prob))
            histp = np.zeros_like(state.histp)
            return histp, prob, fe

        def postupdate_free_energy(nn, fe):
            params = pack(nn.params, layout)
            fe = nn.std * model.apply(params, inputs).reshape(fe.shape)
            return fe - fe.min()

    else:

        def preprocess(_, dy):
            dy, s = normalize_gradients(dy)
            return dy / s, s

        def preupdate_free_energy(state):
            return state.histp, state.prob, state.fe

        def postupdate_free_energy(_, fe):
            return fe

    def train(nn, data):
        targets, s = preprocess(*data)
        params = fit(nn.params, inputs, targets).params
        return NNData(params, nn.mean, s)

    def skip_learning(state):
        return state.histp, state.prob, state.fe, state.nn

    def learn_free_energy(state):
        force = state.Fsum / np.maximum(1, state.hist.reshape(shape))
        histp, prob, fe = preupdate_free_energy(state)

        nn = train(state.nn, (fe, force))
        fe = postupdate_free_energy(nn, fe)

        return histp, prob, fe, nn

    def _learn_free_energy(state, in_training_step):
        return cond(in_training_step, learn_free_energy, skip_learning, state)

    return _learn_free_energy


@dispatch
def build_force_estimator(method: Sirens):
    """
    Returns a function that given a `PartialSirensState` computes an estimate to the force.
    """
    # Local aliases
    f32 = np.float32
    f64 = np.float64

    N = method.N
    grid = method.grid
    model = method.model
    _, layout = unpack(model.parameters)

    model_grad = grad(lambda p, x: model.apply(p, x).sum(), argnums=-1)

    def average_force(state):
        i = state.ind
        return state.Fsum[i] / np.maximum(N, state.hist[i])

    def predict_force(state):
        nn = state.nn
        x = state.xi
        params = pack(nn.params, layout)
        return nn.std * f64(model_grad(params, f32(x)).flatten())

    def _estimate_force(state):
        return cond(state.pred, predict_force, average_force, state)

    if method.restraints is None:
        estimate_force = _estimate_force
    else:
        lo, hi, kl, kh = method.restraints

        def restraints_force(state):
            xi = state.xi.reshape(grid.shape.size)
            return apply_restraints(lo, hi, kl, kh, xi)

        def estimate_force(state):
            ob = np.any(np.array(state.ind) == grid.shape)
            return cond(ob, restraints_force, _estimate_force, state)

    return estimate_force


def _build_gradient_normalizer(grid):
    if grid.is_periodic:

        def normalize_gradients(data):
            axes = tuple(range(data.ndim - 1))
            mean = data.mean(axis=axes)
            std = data.std(axis=axes).max()
            return data - mean, std

    else:

        def normalize_gradients(data):
            axes = tuple(range(data.ndim - 1))
            s = data.std(axis=axes).max()
            return data, s

    return normalize_gradients


@dispatch
def analyze(result: Result[Sirens]):
    """
    Parameters
    ----------

    result: Result[Sirens]
        Result bundle containing the method, final states, and callbacks.

    Returns
    -------

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

    nn: NNData
        Parameters of the resulting trained neural network.

    fnn: NNData
        Parameters of the resulting trained force-based neural network.

    fes_fn: Callable[[JaxArray], JaxArray]
        Function that allows to interpolate the free energy in the CV domain defined
        by the grid.
    """
    method = result.method

    grid = method.grid
    mesh = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    model = method.model
    _, layout = unpack(model.parameters)

    def average_forces(hist, Fsum):
        hist = hist.reshape(*Fsum.shape[:-1], 1)
        return Fsum / np.maximum(hist, 1)

    def build_fes_fn(nn):
        def fes_fn(x):
            params = pack(nn.params, layout)
            A = nn.std * model.apply(params, x) + nn.mean
            return A.max() - A

        return jit(fes_fn)

    histograms = []
    mean_forces = []
    free_energies = []
    nns = []
    fes_fns = []

    # We transpose the data for convenience when plotting
    transpose = grid_transposer(grid)
    d = mesh.shape[-1]

    for s in result.states:
        fes_fn = build_fes_fn(s.nn)
        histograms.append(s.hist)
        mean_forces.append(transpose(average_forces(s.hist, s.Fsum)))
        free_energies.append(transpose(fes_fn(mesh)))
        nns.append(s.nn)
        fes_fns.append(fes_fn)

    ana_result = {
        "histogram": first_or_all(histograms),
        "mean_force": first_or_all(mean_forces),
        "free_energy": first_or_all(free_energies),
        "mesh": transpose(mesh).reshape(-1, d).squeeze(),
        "nn": first_or_all(nns),
        "fes_fn": first_or_all(fes_fns),
    }

    return numpyfy_vals(ana_result)
