# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Utilities for saving and loading the results of `pysages` simulations.

This module provides two functions for managing the persistent storage of a `pysages`
simulation's state using pickle serialization (via the `dill` library).

* `load(filename)`: Attempts to load the simulation state from a file and return
  the corresponding `Result` object.
* `save(result, filename)`: Saves the given `Result` object to a file.

**Note:**

These functions assume pickle's `DEFAULT_PROTOCOL` and data format. Use them with caution
if modifications have been made to the saved data structures.
"""

import dill as pickle

from pysages.backends.snapshot import Box, Snapshot
from pysages.methods import Metadynamics, Unbias
from pysages.methods.core import GriddedSamplingMethod, Result
from pysages.typing import Callable
from pysages.utils import dispatch, identity


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.endswith("snapshot") and name == "Snapshot":
            return _recreate_snapshot
        if module.endswith("dill") and name == "_create_namedtuple":
            # `dill` handles types that inherit from NamedTuple this way
            return _recreate_namedutple
        return super().find_class(module, name)


class _CompatSnapshot:
    def __new__(cls, *args, **kwargs):
        return _recreate_snapshot(*args, **kwargs)


def load(filename) -> Result:
    """
    Loads the state of an previously run `pysages` simulation from a file.

    This function attempts to load the pickled data (via the `dill` library) from a file
    with the given `filename` and return the corresponding `Result` object.

    Parameters
    ----------

    filename: str
        The name of the file containing the pickled data.

    **Notes:**

    This function attempts to maintain backwards compatibility with serialized data
    structures that have changed in different `pysages` versions.
    """
    with open(filename, "rb") as f:
        unpickler = CompatUnpickler(f)
        result = unpickler.load()

    if not isinstance(result, Result):
        raise TypeError("Only loading of `Result` objects is supported.")

    # Update results with `ncalls` estimates for each state
    update_ncalls = _ncalls_estimator(result.method)
    result.states = [update_ncalls(state) for state in result.states]

    return result


def save(result: Result, filename) -> None:
    """
    Saves the result of a `pysages` simulation to a file.

    This function saves the given `Result` object to a file with the specified `filename`
    using pickle serialization (via the `dill` library).

    Parameters
    ----------

    result: Result
        The `Result` object to be saved.

    filename: str
        The name of the file to save the data to.
    """
    with open(filename, "wb") as io:
        pickle.dump(result, io)


@dispatch
def _ncalls_estimator(_) -> Callable:
    # Fallback case. We leave ncalls as zero.
    return identity

@dispatch
def _ncalls_estimator(_: Unbias) -> Callable:
    # Fallback case. We leave ncalls as zero.
    return identity

@dispatch
def _ncalls_estimator(_: Metadynamics) -> Callable:
    def update(state):
        ncalls = state.idx  # use the number of gaussians deposited as proxy
        return state._replace(ncalls=ncalls)

    return update


@dispatch
def _ncalls_estimator(_: GriddedSamplingMethod) -> Callable:
    def update(state):
        ncalls = state.hist.sum().item()  # use the histograms total count as proxy
        return state._replace(ncalls=ncalls)

    return update


def _recreate_namedutple(name, fieldnames, modulename, defaults=None):
    if modulename.endswith("snapshot") and name == "Snapshot":
        return _CompatSnapshot
    return pickle._dill._create_namedtuple(name, fieldnames, modulename, defaults=defaults)


@dispatch
def _recreate_snapshot(*args, **kwargs):
    # Fallback case: just pass the arguments to the constructor.
    return Snapshot(*args, **kwargs)


@dispatch
def _recreate_snapshot(positions, vel_mass, forces, ids, images, box: Box, dt):
    # Older form: `images` argument was required and preceded `box`.
    _extras = () if images is None else (dict(images=images),)
    return Snapshot(positions, vel_mass, forces, ids, box, dt, *_extras)
