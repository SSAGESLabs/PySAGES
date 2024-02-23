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

from pysages.methods import Metadynamics
from pysages.methods.core import GriddedSamplingMethod, Result
from pysages.utils import dispatch


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

    This function attempts to recover from deserialization errors related to missing
    `ncalls` attributes, assuming there is only one state in the file. Older
    multiple-replica simulations (e.g., `UmbrellaSampling`, `SplineString`) cannot be
    recovered using this method.
    """
    with open(filename, "rb") as io:
        bytestring = io.read()

    try:
        return pickle.loads(bytestring)

    except TypeError as e:  # pylint: disable=W0718
        if "ncalls" not in getattr(e, "message", repr(e)):
            raise e

        # We know that states preceed callbacks and we assume there's only one state.
        # Unfortunately, mutiple replica simulations can not be recovered this way, this
        # includes UmbrellaSampling and SplineString for which a different workaround
        # would be needed.
        boundary = b"t\x94\x81\x94a\x8c\tcallbacks\x94]\x94"
        i = bytestring.find(boundary)
        # We add a zero as the number of ncalls and adjust it later
        bytestring = bytestring[:i] + b"K\x00" + bytestring[i:]
        # Try to deserialize again
        result = pickle.loads(bytestring)

        return _estimate_ncalls(result.method, result)


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
def _estimate_ncalls(_, result) -> Result:
    # Fallback case. We leave ncalls as zero.
    return result


@dispatch
def _estimate_ncalls(_: Metadynamics, result) -> Result:
    state = result.states[0]
    ncalls = state.idx  # use the number of gaussians deposited as proxy
    result.states[0] = state._replace(ncalls=ncalls)
    return result


@dispatch
def _estimate_ncalls(_: GriddedSamplingMethod, result) -> Result:
    state = result.states[0]
    ncalls = state.hist.sum().item()  # use the histograms total count as proxy
    result.states[0] = state._replace(ncalls=ncalls)
    return result
