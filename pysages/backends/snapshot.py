# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jaxlib.xla_extension import DeviceArray as JaxArray
from plum import dispatch
from typing import NamedTuple, Union
from pysages.utils import copy

import jax.numpy as np


class Box(NamedTuple("Box", [
    ("H",      JaxArray),
    ("origin", JaxArray),
])):
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
    vel_mass:  Union[tuple, JaxArray]
    forces:    JaxArray
    ids:       JaxArray
    box:       Box
    dt:        Union[float, JaxArray]

    def __repr__(self):
        return "PySAGES " + type(self).__name__


@dispatch
def copy(s: Box):
    return Box(*(copy(x) for x in s))


@dispatch
def copy(s: Snapshot):
    return Snapshot(*(copy(x) for x in s))
