# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collection of helpful classes for methods.

This includes callback functor objects (callable classes).
"""

import copy
from concurrent.futures import Executor, Future

import numpy
from jax import numpy as np
from plum import Dispatcher

# We use this to dispatch on the different `run` implementations
# for `SamplingMethod`s.
methods_dispatch = Dispatcher()


class SerialExecutor(Executor):
    """
    Subclass of `concurrent.futures.Executor` used as the default
    task manager. It will execute all tasks in serial.
    """

    def submit(self, fn, *args, **kwargs):  # pylint: disable=arguments-differ
        """
        Executes `fn(*args, **kwargs)` and returns a `Future` object wrapping the result.
        """
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


class ReplicasConfiguration:
    """
    Stores the information necessary to execute multiple simulation runs,
    including the number of copies of the system and the task manager.
    """

    def __init__(self, copies: int = 1, executor=SerialExecutor()):
        """
        ReplicasConfiguration constructor.

        Parameters
        ----------
        copies: int
            Number of replicas of the simulation system to be generated.
            Defaults to `1`.

        executor:
            Task manager that satisfies the `concurrent.futures.Executor` interface.
            Defaults to `SerialExecutor()`.
        """
        self.copies = copies
        self.executor = executor


class HistogramLogger:
    """
    Implements a Callback functor for methods.
    Logs the state of the collective variable to generate histograms.


    Parameters
    ----------
    period:
        Time steps between logging of collective variables.

    offset:
        Time steps at the beginning of a run used for equilibration.
    """

    def __init__(self, period: int, offset: int = 0):
        self.period = period
        self.counter = 0
        self.offset = offset
        self.data = None

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        self.counter += 1
        if self.counter > self.offset and self.counter % self.period == 0:
            xi = state.xi[0]
            if self.data is None:
                self.data = copy.copy(xi)
            else:
                self.data = np.vstack((self.data, xi))

    def get_histograms(self, **kwargs):
        """
        Helper function to generate histograms from the collected CV data.
        `kwargs` are passed on to `numpy.histogramdd` function.
        """
        data = np.asarray(self.data)
        if "density" not in kwargs:
            kwargs["density"] = True
        return np.histogramdd(data, **kwargs)

    def get_means(self):
        """
        Returns mean values of the histogram data.
        """
        data = np.asarray(self.data)
        return np.mean(data, axis=0)

    def get_cov(self):
        """
        Returns covariance matrix of the histogram data.
        """
        data = np.asarray(self.data)
        return np.cov(data.T)

    def reset(self):
        """
        Reset internal state.
        """
        self.counter = 0
        self.data = None

    def numpyfy(self):
        self.data = numpy.asarray(self.data)


# NOTE: for OpenMM; issue #16 on openmm-dlext should be resolved for this to work properly.
class MetaDLogger:
    """
    Logs the state of the collective variable and other parameters in Metadynamics.

    Parameters
    ----------
    hills_file:
        Name of the output hills log file.

    log_period:
        Time steps between logging of collective variables and Metadynamics parameters.
    """

    def __init__(self, hills_file, log_period):
        """
        MetaDLogger constructor.
        """
        self.hills_file = hills_file
        self.log_period = log_period
        self.counter = 0

    def save_hills(self, xi, sigma, height):
        """
        Append the centers, standard deviations and heights to log file.
        """
        with open(self.hills_file, "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            f.write("\t".join(map(str, xi.flatten())) + "\t")
            f.write("\t".join(map(str, sigma.flatten())) + "\t")
            f.write(str(height) + "\n")

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            idx = state.idx - 1 if state.idx > 0 else 0
            self.save_hills(state.centers[idx], state.sigmas, state.heights[idx])

        self.counter += 1


class Funnel_MetadLogger:
    """
    Logs the state of the funnel_cv the perpendicular cv and other parameters in Funnel_Metadynamics.
    Parameters
    ----------
    hills_file:
        Name of the output hills log file.
    log_period:
        Time steps between logging of collective variables and Metadynamics parameters.
    """

    def __init__(self, hills_file, log_period):
        """
        Funnel_MetaDLogger constructor.
        """
        self.hills_file = hills_file
        self.log_period = log_period
        self.counter = 0

    def save_hills(self, xi, sigma, height, perp):
        """
        Append the centers, sigmas, heights, and perp_cv to log file.
        """
        with open(self.hills_file, "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            f.write("\t".join(map(str, xi.flatten())) + "\t")
            f.write("\t".join(map(str, sigma.flatten())) + "\t")
            f.write(str(height) + "\t")
            f.write(str(perp) + "\n")

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            idx = state.idx - 1 if state.idx > 0 else 0
            self.save_hills(state.centers[idx], state.sigmas, state.heights[idx], state.perp)

        self.counter += 1


class Funnel_Logger:
    """
    Logs the state of the collective variable and other parameters in Funnel.
    Parameters
    ----------
    funnel_file:
        Name of the output funnel log file.
    log_period:
        Time steps between logging of collective variables and Funnel parameters.
    """

    def __init__(self, funnel_file, log_period):
        """
        Funnel_Logger constructor.
        """
        self.funnel_file = funnel_file
        self.log_period = log_period
        self.counter = 0

    def save_work(self, xi, proj, restr):
        """
        Append the funnel_cv, perp_funnel, and funnel_restraints to log file.
        """
        with open(self.funnel_file, "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            f.write("\t".join(map(str, xi.flatten())) + "\t")
            f.write("\t".join(map(str, restr.flatten())) + "\t")
            f.write(str(proj) + "\n")

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            self.save_work(state.xi, state.proj, state.restr)

        self.counter += 1


def listify(arg, replicas, name, dtype):
    """
    Returns a list of with length `replicas` of `arg` if `arg` is not a list,
    or `arg` if it is already a list of length `replicas`.
    """
    if isinstance(arg, list):
        if len(arg) != replicas:
            raise RuntimeError(
                f"Invalid length for argument {name} (got {len(arg)}, expected {replicas})"
            )
        return arg

    return [dtype(arg) for i in range(replicas)]


def numpyfy_vals(dictionary: dict, numpy_only: bool = False):
    """
    Iterate all keys of the dictionary and convert every possible value into a numpy array.
    We recommend to pickle final analyzed results are numpyfying
    with `numpy_only=True` to avoid pickling issues.

    Strings and numpy arrays, that would result in `dtype == object` are not converted.

    Parameters
    ----------

    dictionary: dict
        Input dictionary, which keys are attempted to be converted to numpy arrays.
    numpy_only: bool
        If true, any not simple numpy array object is excluded from the results.
    Returns
    -------

    dict: The same dictionary, but keys are preferably numpy arrays.
    """

    new_dict = {}
    for key in dictionary:
        if not numpy_only:
            new_dict[key] = dictionary[key]
        if isinstance(dictionary[key], str):
            numpy_array = numpy.asarray(dictionary[key])
            if numpy_array.dtype != numpy.dtype("O"):
                new_dict[key] = numpy_array
    return new_dict
