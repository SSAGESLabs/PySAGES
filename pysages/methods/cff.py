# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Combined Force-Frequency (CFF) Sampling

CFF is a free energy sampling method which uses artificial neural networks to generate an
on-the-fly adaptive bias capable of rapidly resolving free energy landscapes. The method
learns both from the frequency of visits to bins in a CV-space and generalized force
estimates. It can be seen as a generalization of ANN and FUNN sampling methods that uses
two neural networks to approximate the free energy and its derivatives.
"""

import numbers
from functools import partial

from jax import jit
from jax import numpy as np
from jax import vmap
from jax.lax import cond

from pysages.approxfun import compute_mesh
from pysages.approxfun import scale as _scale
from pysages.grids import build_indexer, grid_transposer
from pysages.methods.core import NNSamplingMethod, Result, generalize
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.ml.models import MLP
from pysages.ml.objectives import L2Regularization, Sobolev1SSE
from pysages.ml.optimizers import LevenbergMarquardt
from pysages.ml.training import NNData, build_fitting_function, convolve, normalize
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.typing import JaxArray, NamedTuple, Tuple
from pysages.utils import dispatch, first_or_all, linear_solver

# Aliases
f32 = np.float32


class CFFState(NamedTuple):
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
    fnn: NNData
    ncalls: int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialCFFState(NamedTuple):
    xi: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    ind: Tuple
    fnn: NNData
    pred: bool


class CFF(NNSamplingMethod):
    """
    Implementation of the sampling method described in
    "Combined Force-Frequency Sampling for Simulation of
    Systems Having Rugged Free Energy Landscapes"
    [J. Chem. Theory Comput. 2020, 16, 3](https://doi.org/10.1021/acs.jctc.9b00883).

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

    kT: float
        Value of `kT` in the same units as the backend internal energy units.

    N: Optional[int] = 500
        Threshold parameter before accounting for the full average of the
        binned generalized mean force.

    train_freq: Optional[int] = 5000
        Training frequency.

    optimizer: Optional[Optimizer] = LevenbergMarquardt(loss=Sobolev1SSE())
        Optimization method used for training based on both frequencies and forces.

    foptimizer: Optional[Optimizer] = LevenbergMarquardt()
        Optimization method used for training based on forces only.

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

    def __init__(self, cvs, grid, topology, kT, **kwargs):
        # kT must be unitless but consistent with the internal unit system of the backend
        assert isinstance(kT, numbers.Real)

        super().__init__(cvs, grid, topology, **kwargs)

        self.kT = kT
        self.N = np.asarray(self.kwargs.get("N", 500))
        self.train_freq = self.kwargs.get("train_freq", 5000)

        dims = grid.shape.size
        scale = partial(_scale, grid=grid)
        regularizer = L2Regularization(1e-4)
        default_optimizer = LevenbergMarquardt(loss=Sobolev1SSE(), reg=regularizer, max_iters=1000)
        default_foptimizer = LevenbergMarquardt(reg=regularizer)
        self.model = MLP(dims, 1, topology, transform=scale)
        self.fmodel = MLP(dims, dims, topology, transform=scale)
        self.optimizer = kwargs.get("optimizer", default_optimizer)
        self.foptimizer = kwargs.get("foptimizer", default_foptimizer)
        self.use_pinv = self.kwargs.get("use_pinv", False)

    def build(self, snapshot, helpers):
        return _cff(self, snapshot, helpers)


def _cff(method: CFF, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    train_freq = method.train_freq
    dt = snapshot.dt

    # Neural network paramters
    ps, _ = unpack(method.model.parameters)
    fps, _ = unpack(method.fmodel.parameters)

    # Helper methods
    tsolve = linear_solver(method.use_pinv)
    get_grid_index = build_indexer(grid)
    learn_free_energy = build_free_energy_learner(method)
    estimate_force = build_force_estimator(method)

    query, dimensionality, to_force_units = helpers

    def initialize():
        dims = grid.shape.size
        natoms = np.size(snapshot.positions, 0)
        gshape = grid.shape if dims > 1 else (*grid.shape, 1)

        xi, _ = cv(query(snapshot))
        bias = np.zeros((natoms, dimensionality()))
        hist = np.zeros(gshape, dtype=np.uint32)
        histp = np.zeros(gshape, dtype=np.uint32)
        prob = np.zeros(gshape)
        fe = np.zeros(gshape)
        Fsum = np.zeros((*grid.shape, dims))
        force = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        nn = NNData(ps, np.array(0.0), np.array(1.0))
        fnn = NNData(fps, np.zeros(dims), np.array(1.0))

        return CFFState(xi, bias, hist, histp, prob, fe, Fsum, force, Wp, Wp_, nn, fnn, 0)

    def update(state, data):
        # During the intial stage, when there are not enough collected samples, use ABF
        ncalls = state.ncalls + 1
        in_training_regime = ncalls > train_freq
        in_training_step = in_training_regime & (ncalls % train_freq == 1)
        histp, prob, fe, nn, fnn = learn_free_energy(state, in_training_step)
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
        histp = histp.at[I_xi].add(1)
        #
        force = estimate_force(PartialCFFState(xi, hist, Fsum, I_xi, fnn, in_training_regime))
        bias = (-Jxi.T @ force).reshape(state.bias.shape)
        #
        return CFFState(xi, bias, hist, histp, prob, fe, Fsum, force, Wp, state.Wp, nn, fnn, ncalls)

    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_learner(method: CFF):
    """
    Returns a function that given a `CFFState` trains the method's neural networks which
    give estimates of the probability density and the gradient of the free energy.
    """
    kT = method.kT
    grid = method.grid
    model = method.model
    fmodel = method.fmodel
    optimizer = method.optimizer
    foptimizer = method.foptimizer

    dims = grid.shape.size
    shape = (*grid.shape, 1)
    inputs = f32((compute_mesh(grid) + 1) * grid.size / 2 + grid.lower)
    smoothing_kernel = f32(blackman_kernel(dims, 7))
    padding = "wrap" if grid.is_periodic else "edge"
    conv = partial(convolve, kernel=smoothing_kernel, boundary=padding)

    _, layout = unpack(model.parameters)
    _, flayout = unpack(fmodel.parameters)
    fit = build_fitting_function(model, optimizer)
    ffit = build_fitting_function(fmodel, foptimizer)

    def vsmooth(y):
        return vmap(conv)(y.T).T

    smooth = conv if dims > 1 else vsmooth

    def preprocess(y, dy):
        axes = tuple(range(dy.ndim - 1))
        dy, dy_mean, dy_std = normalize(dy, axes=axes)
        s = np.maximum(y.std(), dy_std.max())
        y = smooth(f32(y / s))
        dy = vsmooth(f32(dy * dy_std / s))
        return y, dy, dy_mean, s

    def train(nn, fnn, data):
        y, dy, f_mean, s = preprocess(*data)
        params = fit(nn.params, inputs, (y, dy)).params
        fparams = ffit(fnn.params, inputs, dy).params
        return NNData(params, nn.mean, s), NNData(fparams, f_mean, s)

    def skip_learning(state):
        return state.histp, state.prob, state.fe, state.nn, state.fnn

    def learn_free_energy(state):
        prob = state.prob + state.histp * np.exp(state.fe / kT)
        fe = kT * np.log(np.maximum(1, prob))
        force = state.Fsum / np.maximum(1, state.hist.reshape(shape))
        histp = np.zeros_like(state.histp)

        nn, fnn = train(state.nn, state.fnn, (fe, force))
        params = pack(nn.params, layout)
        fe = nn.std * model.apply(params, inputs).reshape(fe.shape)
        fe = fe - fe.min()

        return histp, prob, fe, nn, fnn

    def _learn_free_energy(state, in_training_step):
        return cond(in_training_step, learn_free_energy, skip_learning, state)

    return _learn_free_energy


@dispatch
def build_force_estimator(method: CFF):
    """
    Returns a function that given a `PartialCFFState` computes an estimate to the force.
    """
    N = method.N
    grid = method.grid
    fmodel = method.fmodel
    _, layout = unpack(fmodel.parameters)

    def average_force(state):
        i = state.ind
        return state.Fsum[i] / np.maximum(N, state.hist[i])

    def predict_force(state):
        fnn = state.fnn
        x = state.xi
        fparams = pack(fnn.params, layout)
        return fnn.std * fmodel.apply(fparams, f32(x)).flatten() + fnn.mean

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


@dispatch
def analyze(result: Result[CFF]):
    """
    Parameters
    ----------

    result: Result[CFF]
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
        nn: NNData
            Parameters of the resulting trained neural network.
        fnn: NNData
            Parameters of the resulting trained force-based neural network.
        fes_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy in the CV domain defined
            by the grid.
    """
    method = result.method
    states = result.states

    grid = method.grid
    mesh = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    model = method.model
    _, layout = unpack(model.parameters)

    def average_forces(hist, Fsum):
        shape = (*Fsum.shape[:-1], 1)
        return Fsum / np.maximum(hist.reshape(shape), 1)

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
    fnns = []
    fes_fns = []

    # We transpose the data for convenience when plotting
    transpose = grid_transposer(grid)
    d = mesh.shape[-1]

    for s in states:
        histograms.append(transpose(s.hist))
        mean_forces.append(transpose(average_forces(s.hist, s.Fsum)))
        free_energies.append(transpose(s.fe.max() - s.fe))
        nns.append(s.nn)
        fnns.append(s.fnn)
        fes_fns.append(build_fes_fn(s.nn))

    ana_result = {
        "histogram": first_or_all(histograms),
        "mean_force": first_or_all(mean_forces),
        "free_energy": first_or_all(free_energies),
        "mesh": transpose(mesh).reshape(-1, d).squeeze(),
        "nn": first_or_all(nns),
        "fnn": first_or_all(fnns),
        "fes_fn": first_or_all(fes_fns),
    }

    return numpyfy_vals(ana_result)
