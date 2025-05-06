# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jax import jit
from jax import numpy as np

from pysages.typing import Callable, JaxArray, NamedTuple, Optional, Tuple, Union
from pysages.utils import copy, dispatch, identity

AbstractBox = NamedTuple("AbstractBox", [("H", JaxArray), ("origin", JaxArray)])


class Box(AbstractBox):
    """
    Simulation box information (origin and transform matrix).
    """

    def __new__(cls, H, origin):
        return super().__new__(cls, np.asarray(H), np.asarray(origin))

    def __repr__(self):
        return "PySAGES " + type(self).__name__


class Snapshot(NamedTuple):
    """
    Stores wrappers around the simulation context information: positions,
    velocities, masses, forces, particles ids, box, and time step size.
    """

    positions: JaxArray
    vel_mass: Union[JaxArray, Tuple[JaxArray, JaxArray]]
    forces: JaxArray
    ids: JaxArray
    images: Optional[JaxArray]
    box: Box
    dt: Union[JaxArray, float]

    def __repr__(self):
        return "PySAGES " + type(self).__name__


class SnapshotMethods(NamedTuple):
    positions: Callable
    indices: Callable
    momenta: Callable
    masses: Callable


class HelperMethods(NamedTuple):
    query: Callable
    dimensionality: Callable[[], int]
    to_force_units: Callable = identity


@dispatch(precedence=1)
def copy(s: Box, *args):
    return Box(*(copy(x, *args) for x in s))


@dispatch(precedence=1)
def copy(s: Snapshot, *args):
    return Snapshot(*(copy(x, *args) for x in s))


# Fallback method for restoring velocities and masses
def restore_vm(view, snapshot, prev_snapshot):
    vel_mass = view(snapshot.vel_mass)
    vel_mass[:] = view(prev_snapshot.vel_mass)


def restore(view, snapshot, prev_snapshot, restore_vm=restore_vm):
    # Create a mutable view of the jax arrays
    positions = view(snapshot.positions)
    forces = view(snapshot.forces)
    ids = view(snapshot.ids)
    # Overwrite the data
    positions[:] = view(prev_snapshot.positions)
    forces[:] = view(prev_snapshot.forces)
    ids[:] = view(prev_snapshot.ids)
    # Special handling for velocities and masses
    restore_vm(view, snapshot, prev_snapshot)
    # Overwrite images if the backend uses them
    if snapshot.images is not None:
        images = view(snapshot.images)
        images[:] = view(prev_snapshot.images)


def build_data_querier(snapshot_methods, flags):
    flags = sorted(flags)
    fields = [(flag, JaxArray) for flag in flags]
    getters = [getattr(snapshot_methods, s) for s in flags]
    ParticleData = NamedTuple("ParticleData", fields)

    def query_data(snapshot):
        return ParticleData(*(p(snapshot) for p in getters))

    return jit(query_data)
