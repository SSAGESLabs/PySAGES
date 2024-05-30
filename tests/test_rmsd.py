#!/usr/bin/env python

import numpy as np
import pytest

from pysages.colvars.orientation import RMSD


@pytest.mark.parametrize(
    "f1, f2, rmsd_expected",
    [
        ("tests/ci2_1.txt", "tests/ci2_2.txt", 11.7768),
        ("tests/ci2_1.txt", "tests/ci2_1t.txt", 0.000493294127),
        ("tests/ci2_2.txt", "tests/ci2_12.txt", 15.422431),
    ],
)
def test_rmsd(f1, f2, rmsd_expected):
    base = np.loadtxt(f1, dtype=str)[:, 6:9].astype(float)
    reference = np.loadtxt(f2, dtype=str)[:, 6:9].astype(float)
    indices = np.arange(len(base))
    rmsd_kabsch = RMSD(indices, reference)
    rmsd_calculated = rmsd_kabsch.function(base)
    rmsd_expected = rmsd_expected
    assert np.allclose(
        rmsd_expected, rmsd_calculated
    ), f"Test Case PDB {f1} {f2} failed: {rmsd_expected, rmsd_calculated}"
