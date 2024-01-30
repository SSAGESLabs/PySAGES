# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collection of analysis methods to compute the free energy from the results of a
biased simulation.
"""

from functools import partial

from jax import jit
from jax import numpy as np
from jax import vmap

from pysages.approxfun import compute_mesh
from pysages.approxfun import scale as _scale
from pysages.grids import grid_transposer
from pysages.methods.core import Result
from pysages.ml.models import MLP
from pysages.ml.objectives import GradientsSSE, L2Regularization
from pysages.ml.optimizers import LevenbergMarquardt
from pysages.ml.training import NNData, build_fitting_function, convolve
from pysages.ml.utils import blackman_kernel, pack, unpack
from pysages.utils import dispatch, first_or_all


class AnalysisStrategy:
    """
    Abstract class for tagging analysis methods that compute free energies.
    """

    pass


class GradientLearning(AnalysisStrategy):
    """
    Tag class for analysis methods that compute free energies by fitting
    the gradient of a neural network.
    """

    pass


@dispatch
def _analyze(result: Result, strategy: GradientLearning, topology):
    """
    Computes the free energy from the result of an `ABF`-based run.
    Integrates the forces via a gradient learning strategy.

    Parameters
    ----------

    result: Result:
        Result bundle containing method, final ABF-like state, and callback.

    strategy: GradientLearning

    topology: Tuple[int]
        Defines the architecture of the neural network
        (number of nodes in each hidden layer).

    Returns
    -------

    dict: A dictionary with the following keys:

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

    NOTE:
    For multiple-replicas runs we return a list (one item per-replica)
    for each attribute.
    """

    # The ForceNN based analysis occurs in two stages:
    #
    #  1. The data is smoothed and a first quick fitting is performed to obtain
    #     an approximate set of network parameters.
    #  2. A second training pass is then performed over the raw data starting
    #     with the parameters from previous step.

    method = result.method
    states = result.states
    grid = method.grid
    mesh = inputs = (compute_mesh(grid) + 1) * grid.size / 2 + grid.lower

    model = MLP(grid.shape.size, 1, topology, transform=partial(_scale, grid=grid))
    loss = GradientsSSE()
    regularizer = L2Regularization(1e-4)

    # Stage 1 optimizer
    pre_optimizer = LevenbergMarquardt(loss=loss, max_iters=250, reg=regularizer)
    pre_fit = build_fitting_function(model, pre_optimizer)

    # Stage 2 optimizer
    optimizer = LevenbergMarquardt(loss=loss, max_iters=1000, reg=regularizer)
    fit = build_fitting_function(model, optimizer)

    @vmap
    def smooth(data, conv_dtype=np.float32):
        data_dtype = data.dtype
        boundary = "wrap" if grid.is_periodic else "edge"
        kernel = np.asarray(blackman_kernel(grid.shape.size, 7), dtype=conv_dtype)
        data = np.asarray(data, dtype=conv_dtype)
        return np.asarray(convolve(data.T, kernel, boundary=boundary), dtype=data_dtype).T

    @jit
    def pre_train(nn, data):
        params = pre_fit(nn.params, inputs, smooth(data)).params
        return NNData(params, nn.mean, nn.std)

    @jit
    def train(nn, data):
        params = fit(nn.params, inputs, data).params
        return NNData(params, nn.mean, nn.std)

    def build_fes_fn(state):
        hist = np.expand_dims(state.hist, state.hist.ndim)
        F = state.Fsum / np.maximum(hist, 1)

        # Scale the mean forces
        s = np.abs(F).max()
        F = F / s

        ps, layout = unpack(model.parameters)
        nn = pre_train(NNData(ps, 0.0, s), F)
        nn = train(nn, F)

        def fes_fn(x):
            params = pack(nn.params, layout)
            A = nn.std * model.apply(params, x) + nn.mean
            return A.max() - A

        return jit(fes_fn)

    def average_forces(hist, Fsum):
        shape = (*Fsum.shape[:-1], 1)
        return Fsum / np.maximum(hist.reshape(shape), 1)

    hists = []
    mean_forces = []
    free_energies = []
    fes_fns = []

    # We transpose the data for convenience when plotting
    transpose = grid_transposer(grid)
    d = mesh.shape[-1]

    for state in states:
        fes_fn = build_fes_fn(state)
        hists.append(transpose(state.hist))
        mean_forces.append(transpose(average_forces(state.hist, state.Fsum)))
        free_energies.append(transpose(fes_fn(mesh)))
        fes_fns.append(fes_fn)

    return {
        "histogram": first_or_all(hists),
        "mean_force": first_or_all(mean_forces),
        "free_energy": first_or_all(free_energies),
        "fes_fn": first_or_all(fes_fns),
        "mesh": transpose(mesh).reshape(-1, d).squeeze(),
    }
