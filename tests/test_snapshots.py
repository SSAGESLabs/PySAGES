from pysages.backends.snapshot import Box, Snapshot
from pysages.utils import ToCPU, copy

import jax.numpy as np
import numpy
import pytest


def test_copying():
    positions = np.ones((2, 3))
    vel_mass = np.zeros((2, 4))
    forces = np.zeros((2, 3))
    ids = np.zeros((2,), dtype = np.uint32)
    images = np.ones((2, 3), dtype=np.int32)
    H = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    origin = (0.0, 0.0, 0.0)
    box = Box(H, origin)
    dt = 0.1

    old = Snapshot(positions, vel_mass, forces, ids, images, box, dt)
    new = copy(old)

    old_ptr = old.positions.unsafe_buffer_pointer()
    new_ptr = new.positions.unsafe_buffer_pointer()
    old_box_H_ptr = old.box.H.unsafe_buffer_pointer()
    new_box_H_ptr = new.box.H.unsafe_buffer_pointer()

    # When copying to CPU we get a `numpy.ndarray` instead of a
    # `jaxlib.xla_extension.DeviceArray`
    new_cpu = copy(old, ToCPU())

    assert np.all(old.positions == new.positions).item()
    assert np.all(old.box.H == new.box.H).item()
    assert old_ptr != new_ptr
    assert old_box_H_ptr != new_box_H_ptr
    assert type(new_cpu.positions) is numpy.ndarray
