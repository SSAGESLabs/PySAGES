#!/usr/bin/env python

from math import atan2, cos, pi, sin, sqrt

import numpy as np

sugar_coords = np.loadtxt("sugar_coords.txt")
pucker_coord = np.loadtxt("phase_angle.txt")
q_pysages = pucker_coord[:, 0]
phi_pysages = pucker_coord[:, 1]


data1 = sugar_coords[:, :3]
data2 = sugar_coords[:, 3:6]
data3 = sugar_coords[:, 6:9]
data4 = sugar_coords[:, 9:12]
data5 = sugar_coords[:, 12:15]

r0 = (data1 + data2 + data3 + data4 + data5) * (1.0 / 5.0)
r1 = data1 - r0
r2 = data2 - r0
r3 = data3 - r0
r4 = data4 - r0
r5 = data5 - r0

R1 = (
    (r1 * sin(2 * pi * 0.0 / 5.0))
    + (r2 * sin(2 * pi * 1.0 / 5.0))
    + (r3 * sin(2 * pi * 2.0 / 5.0))
    + (r4 * sin(2 * pi * 3.0 / 5.0))
    + (r5 * sin(2 * pi * 4.0 / 5.0))
)

R2 = (
    (r1 * cos(2 * pi * 0.0 / 5.0))
    + (r2 * cos(2 * pi * 1.0 / 5.0))
    + (r3 * cos(2 * pi * 2.0 / 5.0))
    + (r4 * cos(2 * pi * 3.0 / 5.0))
    + (r5 * cos(2 * pi * 4.0 / 5.0))
)

x = np.cross(R1[0], R2[0])
n = x / sqrt(pow(x[0], 2) + pow(x[1], 2) + pow(x[2], 2))
r1_d = np.dot(r1, n)
r2_d = np.dot(r2, n)
r3_d = np.dot(r3, n)
r4_d = np.dot(r4, n)
r5_d = np.dot(r5, n)
D = (
    (
        (r1_d * sin(4 * pi * 0.0 / 5.0))
        + (r2_d * sin(4 * pi * 1.0 / 5.0))
        + (r3_d * sin(4 * pi * 2.0 / 5.0))
        + (r4_d * sin(4 * pi * 3.0 / 5.0))
        + (r5_d * sin(4 * pi * 4.0 / 5.0))
    )
    * -1
    * sqrt(2.0 / 5.0)
)
C = (
    (r1_d * cos(4 * pi * 0.0 / 5.0))
    + (r2_d * cos(4 * pi * 1.0 / 5.0))
    + (r3_d * cos(4 * pi * 2.0 / 5.0))
    + (r4_d * cos(4 * pi * 3.0 / 5.0))
    + (r5_d * cos(4 * pi * 4.0 / 5.0))
) * sqrt(2.0 / 5.0)

phi_calc = np.arctan2(D, C)
q_calc = np.sqrt(D**2 + C**2)
assert (
    np.mean((q_pysages - q_calc) ** 2) < 1e-4
), "the difference between pysages phase angle and np version is too large!"
assert (
    np.mean((phi_pysages - phi_calc) ** 2) < 1e-4
), "the difference between pysages amplitude and np version is too large!"
print("cheking for pucker pucker passed!")
