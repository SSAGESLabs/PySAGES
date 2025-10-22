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

import io
import dill as pickle

from pysages.backends.snapshot import Snapshot, _migrate_old_snapshot
from pysages.methods import Metadynamics
from pysages.methods.core import GriddedSamplingMethod, Result
from pysages.typing import Callable
from pysages.utils import dispatch, identity


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
    `ncalls` attributes.
    """
    with open(filename, "rb") as io:
        bytestring = io.read()

    try:
        return pickle.loads(bytestring)

    except (TypeError, AttributeError) as e:  # pylint: disable=W0718
        # Handle both ncalls and Snapshot format migration
        error_msg = getattr(e, "message", repr(e))
        
        if "ncalls" in error_msg:
            # Handle ncalls migration (existing logic)
            pass
        elif "Snapshot" in error_msg or "images" in error_msg:
            # Handle Snapshot format migration
            return _handle_snapshot_migration(bytestring)
        else:
            raise e

        # We know that states preceed callbacks so we try to find all tuples of values
        # corresponding to each state.
        j = bytestring.find(b"\x8c\x06states\x94")
        k = bytestring.find(b"\x8c\tcallbacks\x94")
        boundary = b"t\x94\x81\x94"

        marks = []
        while True:
            i = j
            j = bytestring.find(boundary, i + 1, k)
            if j == -1:
                marks.append((i, len(bytestring)))
                break
            marks.append((i, j))

        # We set `ncalls` as zero and adjust it later
        first = marks[0]
        last = marks.pop()
        slices = [
            bytestring[: first[0]],
            *(bytestring[i:j] + b"K\x00" for (i, j) in marks),
            bytestring[last[0] :],  # noqa: E203
        ]
        bytestring = b"".join(slices)

        # Try to deserialize again
        result = pickle.loads(bytestring)

        # Update results with `ncalls` estimates for each state
        update = _ncalls_estimator(result.method)
        result.states = [update(state) for state in result.states]

        return result


def _handle_snapshot_migration(bytestring):
    """
    Handle migration of old Snapshot format during deserialization.
    
    This function attempts to deserialize data that contains old Snapshot
    objects and migrate them to the new format.
    """
    # Create a custom unpickler that can handle Snapshot migration
    class SnapshotMigrationUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Intercept Snapshot class loading
            if name == "Snapshot" and module.endswith("snapshot"):
                return _create_migrating_snapshot_class()
            return super().find_class(module, name)
    
    def _create_migrating_snapshot_class():
        """Create a class that can handle both old and new Snapshot formats."""
        class MigratingSnapshot:
            def __new__(cls, *args, **kwargs):
                # If called with old format, migrate it
                if len(args) == 7:  # old format: (positions, vel_mass, forces, ids, images, box, dt)
                    return _migrate_old_snapshot(args)
                elif len(args) == 6:  # new format: (positions, vel_mass, forces, ids, box, dt, extras)
                    return Snapshot(*args)
                else:
                    return Snapshot(*args, **kwargs)
        
        return MigratingSnapshot
    
    try:
        unpickler = SnapshotMigrationUnpickler(io.BytesIO(bytestring))
        return unpickler.load()
    except Exception:
        # If migration fails, try the original approach
        return pickle.loads(bytestring)


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
