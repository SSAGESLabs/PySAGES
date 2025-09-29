# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Adaptive Enhanced Sampling by Force-biasing Using Neural Networks (FUNN).

FUNN learns the generalized mean forces as a function of some collective variables,
by training a neural network from a binned estimate of the mean forces estimates.
It closely follows the Adaptive Biasing Force (ABF) algorithm, except for biasing
the simulation, which is done from the continuous approximation to the
generalized mean force provided by the network.

To find the free energy surface, it is necessary to integrate the forces by some
appropriate method.
"""

from functools import partial

from jax import jit
from jax import numpy as np
from jax import vmap
from jax.lax import cond

from pysages.approxfun import compute_mesh
from pysages.approxfun import scale as _scale
from pysages.grids import build_indexer
from pysages.methods.analysis import GradientLearning, _analyze
from pysages.methods.core import NNSamplingMethod, Result, generalize
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.ml.models import MLP
from pysages.ml.objectives import L2Regularization
from pysages.ml.optimizers import LevenbergMarquardt
from pysages.ml.training import NNData, build_fitting_function, convolve, normalize
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.typing import JaxArray, NamedTuple, Tuple
from pysages.utils import dispatch, first_or_all, linear_solver


class FUNNState(NamedTuple):
    """
    FUNN internal state.

    Parameters
    ----------

    xi: JaxArray (CV shape)
        Last collective variable recorded in the simulation.

    bias: JaxArray (natoms, 3)
        Array with biasing forces for each particle.

    hist: JaxArray (grid.shape)
        Histogram of visits to the bins in the collective variable grid.

    Fsum: JaxArray (grid.shape, CV shape)
        The cumulative force recorded at each bin of the CV grid.

    Wp: JaxArray (CV shape)
        Estimate of the product $W p$ where `p` is the matrix of momenta and
        `W` the Moore-Penrose inverse of the Jacobian of the CVs.

    Wp_: JaxArray (CV shape)
        The value of `Wp` for the previous integration step.

    nn: NNData
        Bundle of the neural network parameters, and output scaling coefficients.

    ncalls: int
        Counts the number of times the method's update has been called.
    """

    xi: JaxArray
    bias: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    F: JaxArray
    Wp: JaxArray
    Wp_: JaxArray
    nn: NNData
    ncalls: int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialFUNNState(NamedTuple):
    xi: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    ind: Tuple
    nn: NNData
    pred: bool


class FUNN(NNSamplingMethod):
    """
    Implementation of the sampling method described in
    "Adaptive enhanced sampling by force-biasing using neural networks"
    [J. Chem. Phys. 148, 134108 (2018)](https://doi.org/10.1063/1.5020733).

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

    N: Optional[int] = 500
        Threshold parameter before accounting for the full average of the
        binned generalized mean force.

    train_freq: Optional[int] = 5000
        Training frequency.

    optimizer: Optional[Optimizer]
        Optimization method used for training, defaults to LevenbergMarquardt().

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
        super().__init__(cvs, grid, topology, **kwargs)

        self.N = np.asarray(kwargs.get("N", 500))
        self.train_freq = kwargs.get("train_freq", 5000)

        # Neural network and optimizer intialization
        dims = grid.shape.size
        scale = partial(_scale, grid=grid)
        self.model = MLP(dims, dims, topology, transform=scale)
        default_optimizer = LevenbergMarquardt(reg=L2Regularization(1e-6))
        self.optimizer = kwargs.get("optimizer", default_optimizer)
        self.use_pinv = self.kwargs.get("use_pinv", False)

    def build(self, snapshot, helpers):
        return _funn(self, snapshot, helpers)


def _funn(method, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    train_freq = method.train_freq

    dt = snapshot.dt
    dims = grid.shape.size
    natoms = np.size(snapshot.positions, 0)

    # Neural network and optimizer
    ps, _ = unpack(method.model.parameters)

    # Helper methods
    tsolve = linear_solver(method.use_pinv)
    get_grid_index = build_indexer(grid)
    learn_free_energy_grad = build_free_energy_grad_learner(method)
    estimate_free_energy_grad = build_force_estimator(method)

    query, dimensionality, to_force_units = helpers

    def initialize():
        xi, _ = cv(query(snapshot))
        bias = np.zeros((natoms, dimensionality()))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        F = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        nn = NNData(ps, F, F)
        return FUNNState(xi, bias, hist, Fsum, F, Wp, Wp_, nn, 0)

    def update(state, data):
        # During the intial stage, when there are not enough collected samples, use ABF
        ncalls = state.ncalls + 1
        in_training_regime = ncalls > 2 * train_freq
        in_training_step = in_training_regime & (ncalls % train_freq == 1)
        # NN training
        nn = learn_free_energy_grad(state, in_training_step)
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        #
        p = data.momenta
        Wp = tsolve(Jxi, p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        #
        I_xi = get_grid_index(xi)
        hist = state.hist.at[I_xi].add(1)
        Fsum = state.Fsum.at[I_xi].add(to_force_units(dWp_dt) + state.F)
        #
        F = estimate_free_energy_grad(
            PartialFUNNState(xi, hist, Fsum, I_xi, nn, in_training_regime)
        )
        bias = (-Jxi.T @ F).reshape(state.bias.shape)
        #
        return FUNNState(xi, bias, hist, Fsum, F, Wp, state.Wp, nn, ncalls)

    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_grad_learner(method: FUNN):
    """
    Returns a function that given a `FUNNState` trains the method's neural network
    parameters from an ABF-like estimate for the gradient of the free energy.

    The training data is regularized by convolving it with a Blackman window.
    """

    grid = method.grid
    dims = grid.shape.size
    model = method.model

    # Training data
    inputs = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    smoothing_kernel = blackman_kernel(dims, 7)
    padding = "wrap" if grid.is_periodic else "edge"
    conv = partial(convolve, kernel=smoothing_kernel, boundary=padding)
    smooth = jit(lambda y: vmap(conv)(y.T).T)

    _, layout = unpack(model.parameters)
    fit = build_fitting_function(model, method.optimizer)

    def train(nn, y):
        axes = tuple(range(y.ndim - 1))
        y, mean, std = normalize(y, axes=axes)
        reference = smooth(y)
        params = fit(nn.params, inputs, reference).params
        return NNData(params, mean, std / reference.std(axis=axes))

    def learn_free_energy_grad(state):
        hist = np.expand_dims(state.hist, state.hist.ndim)
        F = state.Fsum / np.maximum(hist, 1)
        return train(state.nn, F)

    def skip_learning(state):
        return state.nn

    def _learn_free_energy_grad(state, in_training_step):
        return cond(in_training_step, learn_free_energy_grad, skip_learning, state)

    return _learn_free_energy_grad


def build_force_estimator(method: FUNN):
    """
    Returns a function that given the neural network parameters and a CV value,
    evaluates the network on the provided CV.
    """
    f32 = np.float32
    f64 = np.float64

    N = method.N
    model = method.model
    grid = method.grid
    _, layout = unpack(model.parameters)

    def average_force(state):
        i = state.ind
        return state.Fsum[i] / np.maximum(N, state.hist[i])

    def predict_force(state):
        nn = state.nn
        x = state.xi
        params = pack(nn.params, layout)
        return nn.std * f64(model.apply(params, f32(x)).flatten()) + nn.mean

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
            ob = np.any(np.array(state.ind) == grid.shape)  # Out of bounds condition
            return cond(ob, restraints_force, _estimate_force, state)

    return estimate_force


@dispatch
def analyze(result: Result[FUNN], **kwargs):
    """
    Parameters
    ----------

    result: Result[FUNN]
        Result bundle containing the method, final states, and callbacks.

    topology: Optional[Tuple[int]] = result.method.topology
        Defines the architecture of the neural network
        (number of nodes in each hidden layer).

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
            Coefficients of the basis functions expansion approximating the free energy.

        fes_fn: Callable[[JaxArray], JaxArray]
            Function that allows to interpolate the free energy in the
            CV domain defined by the grid.

    NOTE:
    For multiple-replicas runs we return a list (one item per-replica)
    for each attribute.
    """
    topology = kwargs.get("topology", result.method.topology)
    _result = _analyze(result, GradientLearning(), topology)
    _result["nn"] = first_or_all([state.nn for state in result.states])
    return numpyfy_vals(_result)
