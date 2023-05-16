# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# Code adapted from
# https://github.com/cgohlke/transformations/blob/v2022.9.26/transformations/transformations.py

from typing import List, NamedTuple

from jax import lax
from jax import numpy as np

from pysages.utils.core import dispatch, eps

# axes indices sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of encodings for:
# (
#     "first axis": {"x": 0, "y": 1, "z": 2},
#     "axes ordering": {"right": 0, "left": 1},
#     "axes sequence": {"Proper Euler": 0, "Tait-Bryan": 1},
#     "frame/rotation": {"static/extrinsic": 0, "rotating/intrinsic": 1}
# )
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


class EulerAnglesType(NamedTuple):
    "Base class for Euler angles types."
    pass


class ProperEuler(EulerAnglesType):
    """
    Dispatch class for signaling
    `Proper Euler angles <https://en.wikipedia.org/wiki/Euler_angles#Proper_Euler_angles>`_.
    """

    pass


class TaitBryan(EulerAnglesType):
    """
    Dispatch class for signaling
    `Tait-Bryan angles <https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles>`_.
    """

    pass


class RotationAxes:
    """
    Handles the translation from string or tuple encodings for rotations to an
    appropritate representation for the `quaternion_from_euler` implementation.

    Parameters
    ----------
    axes: Union[str, truple]
        One of 24 axis sequences as string or encoded tuple
    """

    class Parameters(NamedTuple):
        sequence: EulerAnglesType
        j_sign: int
        permutation: List[int]
        intrinsic: bool

    def __init__(self, axes):
        self.params = self.process_axes(axes)

    @dispatch
    def process_axes(self, axes: str):
        return self.process_axes(_AXES2TUPLE[axes.lower()], validate=False)

    @dispatch
    def process_axes(self, rotation_mode: tuple, validate=True):  # noqa: F811
        if validate:
            _TUPLE2AXES[rotation_mode]

        first_axis, left_ordering, proper_euler, intrinsic = rotation_mode

        j_sign = -1 if left_ordering else 1
        sequence = ProperEuler() if proper_euler else TaitBryan()

        o = left_ordering
        i = first_axis + 1
        j = _NEXT_AXIS[i + o - 1] + 1
        k = _NEXT_AXIS[i - o] + 1
        invperm = (0, i, j, k)
        permutation = [invperm[n] for n in invperm]

        return self.Parameters(sequence, j_sign, permutation, intrinsic)


def quaternion_from_euler(ai, aj, ak, axes=RotationAxes("sxyz")):
    """
    Return a quaternion from Euler angles and axis sequence.

    Arguments
    ---------
    ai, aj, ak: numbers.Real
        Euler's roll, pitch and yaw angles

    axes: RotationAxes
        One of 24 axis sequences as string or tuple wrapped as a RotationAxes
    """

    @dispatch
    def quaternion_entries(seq: ProperEuler, cj, sj, cc, ss, cs, sc, sgn):
        v0 = cj * (cc - ss)
        vi = cj * (cs + sc)
        vj = sj * (cc + ss)
        vk = sj * (cs - sc)
        return (v0, vi, sgn * vj, vk)

    @dispatch
    def quaternion_entries(seq: TaitBryan, cj, sj, cc, ss, cs, sc, sgn):  # noqa: F811
        v0 = cj * cc + sj * ss
        vi = cj * sc - sj * cs
        vj = cj * ss + sj * cc
        vk = cj * cs - sj * sc
        return (v0, vi, sgn * vj, vk)

    def _quaternion_from_euler(ai, aj, ak, sequence, j_sign, permutation):
        ai /= 2.0
        aj /= 2.0
        ak /= 2.0
        ci = np.cos(ai)
        si = np.sin(ai)
        cj = np.cos(aj)
        sj = np.sin(aj)
        ck = np.cos(ak)
        sk = np.sin(ak)
        cc = ci * ck
        cs = ci * sk
        sc = si * ck
        ss = si * sk

        v = quaternion_entries(sequence, cj, sj, cc, ss, cs, sc, j_sign)
        return np.array([v[i] for i in permutation])

    sequence, j_sign, permutation, intrinsic = axes.params
    angles = (ak, j_sign * aj, ai) if intrinsic else (ai, j_sign * aj, ak)
    return _quaternion_from_euler(*angles, sequence, j_sign, permutation)


def quaternion_matrix(quaternion, dtype: type = np.zeros(0).dtype):
    """
    Return the homogeneous rotation matrix from a quaternion.
    """

    def _identity_matrix(*_):
        return np.identity(4, dtype=dtype)

    def _quaternion_matrix(q, n):
        q *= np.sqrt(2.0 / n)
        Q = np.outer(q, q)
        return np.array(
            [
                [1.0 - Q[2, 2] - Q[3, 3], Q[1, 2] - Q[3, 0], Q[1, 3] + Q[2, 0], 0.0],
                [Q[1, 2] + Q[3, 0], 1.0 - Q[1, 1] - Q[3, 3], Q[2, 3] - Q[1, 0], 0.0],
                [Q[1, 3] - Q[2, 0], Q[2, 3] + Q[1, 0], 1.0 - Q[1, 1] - Q[2, 2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    q = np.array(quaternion, dtype=dtype)
    n = np.dot(q, q)

    return lax.cond(n < 4 * eps(), _identity_matrix, _quaternion_matrix, q, n)
