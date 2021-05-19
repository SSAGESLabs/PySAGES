# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)
# Copyright (c) 2021: Ludwig Schneider (see LICENSE.md)

import jax.numpy as np
from jax.numpy import linalg
from .core import ThreePointCV, FourPointCV


class Angle(ThreePointCV):
    @property
    def function(self):
        return angle


def angle(p1, p2, p3):
    """
    Returns the angle defined by three points in space
    (around the one in the middle).
    """
    q = p1 - p2
    r = p3 - p2
    return np.arctan2(linalg.norm(np.cross(q, r)), np.dot(q, r))


class DihedralAngle(FourPointCV):
    @property
    def function(self):
        return dihedral_angle


def dihedral_angle(p1, p2, p3, p4):
    """
    Returns the dihedral angle defined by four points in space
    (around the line defined by the two central points).
    """
    q = p3 - p2
    r = np.cross(p2 - p1, q)
    s = np.cross(q, p4 - p3)
    return np.arctan2(np.dot(np.cross(r, s), q), np.dot(r, s) * linalg.norm(q))
