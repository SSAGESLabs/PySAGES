from typing import NamedTuple

from jax import numpy as np

from pysages.colvars import Distance
from pysages.colvars.core import build
from pysages.utils import JaxArray


POSITIONS = np.array(
    [
        [0.0929926, 0.966452, 0.424666],
        [0.415969, 0.485482, 0.408579],
    ]
)


class Snapshot(NamedTuple):
    positions: JaxArray = POSITIONS
    indices: JaxArray = np.arange(len(POSITIONS))


SNAPSHOT = Snapshot()


def test_groups():
    # Verify that we get the same result when using groups
    cv = Distance([0, 1])
    f = build(cv, diff=False)
    assert len(cv.groups) == 0
    assert np.isclose(f(SNAPSHOT).item(), 0.57957285)
    cv = Distance([[0], [1]])
    f = build(cv, diff=False)
    assert len(cv.groups) == 2
    assert np.isclose(f(SNAPSHOT).item(), 0.57957285)
