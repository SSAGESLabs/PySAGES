# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Artificial Neural Network (ANN) sampling.

ANN estimates the probability distribution as a function of a set of collective
variables from the frequency of visits to each bin in a grid in CV space.
Periodically, a binned estimate of the free energy (computed from
the probability density estimate) is used to train a neural network that
provides a continuous approximation to the free energy.
The gradient of the neural network model with respect to the CVs is then used
as biasing force for the simulation.
"""

from functools import partial
from typing import NamedTuple

from jax import grad, numpy as np, vmap
from jax.lax import cond

from pysages.approxfun import compute_mesh, scale as _scale
from pysages.grids import build_indexer
from pysages.methods.core import NNSamplingMethod, generalize
from pysages.ml.models import MLP
from pysages.ml.objectives import L2Regularization
from pysages.ml.optimizers import LevenbergMarquardt
from pysages.ml.training import (
    NNData,
    build_fitting_function,
    normalize,
    convolve,
)
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.utils import Int, JaxArray, dispatch


class ANNState(NamedTuple):
    """
    ANN internal state.

    Parameters
    ----------

    xi : JaxArray (CV shape)
        Last collective variable recorded in the simulation.

    bias : JaxArray (natoms, 3)
        Array with biasing forces for each particle.

    hist: JaxArray (grid.shape)
        Histogram of visits to the bins in the collective variable grid.

    phi: JaxArray (grid.shape, CV shape)
        The current estimate of the free energy.

    prob: JaxArray (CV shape)
        The current estimate of the unnormalized probability distribution.

    nn: NNDada
        Bundle of the neural network parameters, and output scaling coefficients.

    nstep: int
        Count the number of times the method's update has been called.
    """

    xi: JaxArray
    bias: JaxArray
    hist: JaxArray
    phi: JaxArray
    prob: JaxArray
    nn: NNData
    nstep: Int

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class ANN(NNSamplingMethod):
    """
    Implementation of the sampling method described in
    "Learning free energy landscapes using artificial neural networks"
    [J. Chem. Phys. 148, 104111 (2018)](https://doi.org/10.1063/1.5018708).
    """

    snapshot_flags = {"positions", "indices"}

    def __init__(self, cvs, grid, topology, kT, **kwargs):
        """
        Arguments
        ---------

        cvs: Union[List, Tuple]
            List of collective variables.

        grid: Grid
            Specifies the CV domain and number of bins for discretizing the CV space
            along each CV dimension.

        topology: Tuple[int]
            Defines the architecture of the neural network
            (number of nodes of each hidden layer).

        kT: float
            Value of kT in the same units as the backend internal energy units.

        Keyword arguments
        -----------------

        train_freq: int = 5000
            Training frequency.
        """
        super().__init__(cvs, grid, topology, **kwargs)

        self.kT = kT
        self.train_freq = kwargs.get("train_freq", 5000)

        # Neural network and optimizer intialization
        scale = partial(_scale, grid=grid)
        self.model = MLP(grid.shape.size, 1, topology, transform=scale)
        default_optimizer = LevenbergMarquardt(reg=L2Regularization(1e-6))
        self.optimizer = kwargs.get("optimizer", default_optimizer)

    def build(self, snapshot, helpers):
        return _ann(self, snapshot, helpers)


def _ann(method: ANN, snapshot, helpers):
    cv = method.cv
    grid = method.grid
    train_freq = method.train_freq

    shape = grid.shape
    shape = shape if shape.size > 1 else (*shape, 1)
    natoms = np.size(snapshot.positions, 0)

    # Initial Neural network intial parameters
    ps, _ = unpack(method.model.parameters)

    # Helper methods
    get_grid_index = build_indexer(grid)
    learn_free_energy = build_free_energy_learner(method)
    estimate_force = build_force_estimator(method)

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, 3))
        hist = np.zeros(shape, dtype=np.uint32)
        phi = np.zeros(shape)
        prob = np.ones(shape)
        nn = NNData(ps, np.array(0.0), np.array(1.0))
        return ANNState(xi, bias, hist, phi, prob, nn, 1)

    def update(state, data):
        nstep = state.nstep
        in_training_regime = nstep > train_freq
        # We only train every `train_freq` timesteps
        in_training_step = in_training_regime & (nstep % train_freq == 1)
        hist, phi, prob, nn = learn_free_energy(state, in_training_step)
        # Compute the collective variable and its jacobian
        xi, Jxi = cv(data)
        I_xi = get_grid_index(xi)
        hist = hist.at[I_xi].add(1)
        F = estimate_force(nn, xi, in_training_regime)
        bias = np.reshape(-Jxi.T @ F, state.bias.shape)
        #
        return ANNState(xi, bias, hist, phi, prob, nn, nstep + 1)

    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_learner(method: ANN):
    """
    Returns a function that given an `ANNState` trains the method's neural network
    parameters from an estimate to the probability density.

    The training data is regularized by convolving it with a Blackman window.
    """

    kT = method.kT
    grid = method.grid
    dims = grid.shape.size
    model = method.model

    # Training data
    inputs = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower
    smoothing_kernel = blackman_kernel(dims, 7)
    padding = "wrap" if grid.is_periodic else "edge"
    conv = partial(convolve, kernel=smoothing_kernel, boundary=padding)
    smooth = conv if dims > 1 else (lambda y: vmap(conv)(y.T).T)

    _, layout = unpack(model.parameters)
    fit = build_fitting_function(model, method.optimizer)

    def train(nn, y):
        y, mean, std = normalize(y)
        reference = smooth(y)
        params = fit(nn.params, inputs, reference).params
        return NNData(params, mean, std / reference.std())

    def learn_free_energy(state):
        prob = state.prob + state.hist * np.exp(state.phi / kT)
        phi = kT * np.log(prob)
        #
        nn = train(state.nn, phi)
        params = pack(nn.params, layout)
        phi = nn.std * model.apply(params, inputs).reshape(phi.shape)
        phi = phi - phi.min()
        #
        hist = np.zeros_like(state.hist)
        #
        return hist, phi, prob, nn

    def skip_learning(state):
        return state.hist, state.phi, state.prob, state.nn

    def _learn_free_energy(state, in_training_step):
        return cond(in_training_step, learn_free_energy, skip_learning, state)

    return _learn_free_energy


@dispatch
def build_force_estimator(method: ANN):
    """
    Returns a function that given the neural network parameters and a CV value,
    computes the gradient of the network (the mean force) with respect to CV.
    """

    f32 = np.float32
    f64 = np.float64

    grid = method.grid
    dims = grid.shape.size
    model = method.model
    _, layout = unpack(model.parameters)

    model_grad = grad(lambda p, x: model.apply(p, x).sum(), argnums=1)

    def predict_force(data):
        nn, x = data
        params = pack(nn.params, layout)
        return nn.std * f64(model_grad(params, f32(x)).flatten())

    def zero_force(data):
        return np.zeros(dims)

    def estimate_force(xi, I_xi, nn, in_training_regime):
        in_bounds = np.all(np.array(I_xi) != grid.shape)
        use_nn = in_training_regime & in_bounds
        return cond(use_nn, predict_force, zero_force, (nn, xi))

    return estimate_force
