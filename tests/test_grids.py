import jax.numpy as np
import pytest
from jax.numpy import pi
from jax.numpy import uint32 as UInt32

from pysages.grids import Chebyshev, Grid, Periodic, Regular, build_indexer, convert

lower_1d = (-pi,)
upper_1d = (pi,)
shape_1d = (64,)

lower_2d = (-pi, -1.0)
upper_2d = (pi, 1.0)
shape_2d = (64, 32)


def test_Grid_constructor():
    lower = lower_1d
    upper = upper_1d
    shape = shape_1d

    # Default constructor
    grid = Grid(lower, upper, shape)
    # Periodic grid kw constructor
    periodic_grid = Grid(lower, upper, shape, periodic=True)
    # Parametric constructors
    grid_tv = Grid[Regular](lower, upper, shape)
    cheb_grid_tv = Grid[Chebyshev](lower, upper, shape)
    periodic_grid_tv = Grid[Periodic](lower, upper, shape)

    assert grid == grid_tv
    assert periodic_grid == periodic_grid_tv
    assert grid != periodic_grid
    assert cheb_grid_tv == convert(grid, Grid[Chebyshev])

    assert ~grid.is_periodic
    assert periodic_grid.is_periodic

    # Test constructor exceptions
    with pytest.raises(TypeError):
        Grid(lower, upper, shape, periodic=None)
    with pytest.raises(TypeError):
        Grid(lower, upper, shape, periodic=1)
    with pytest.raises(TypeError):
        Grid[1](lower, upper, shape)
    with pytest.raises(TypeError):
        Grid[bool](lower, upper, shape)
    with pytest.raises(ValueError):
        Grid(lower, upper, shape, invalid_keyword=True)
    with pytest.raises(ValueError):
        Grid[Periodic](lower, upper, shape, periodic=False)
    with pytest.raises(ValueError):
        Grid[Chebyshev](lower, upper, shape, periodic=True)


def test_grid_indexing():
    lower = lower_1d
    upper = upper_1d
    shape = shape_1d

    grid = Grid(lower, upper, shape)
    cheb_grid = Grid[Chebyshev](lower, upper, shape)
    periodic_grid = Grid[Periodic](lower, upper, shape)

    get_index = build_indexer(grid)
    cheb_get_index = build_indexer(cheb_grid)
    periodic_get_index = build_indexer(periodic_grid)

    # Indexing 1D
    assert get_index(-pi - 1) == (UInt32(64),)
    assert get_index(-pi) == (UInt32(0),)
    assert get_index(0.0) == (UInt32(32),)
    assert cheb_get_index(-pi - 1) == (UInt32(64),)
    assert cheb_get_index(-pi) == (UInt32(0),)
    assert cheb_get_index(-pi * 0.996) == (UInt32(1),)
    assert cheb_get_index(0.0) == (UInt32(32),)
    assert periodic_get_index(-pi * (1 + 0.99 / 32)) == (UInt32(63),)
    assert periodic_get_index(-pi) == periodic_get_index(pi) == (UInt32(0),)
    assert periodic_get_index(0.0) == (UInt32(32),)

    lower = lower_2d
    upper = upper_2d
    shape = shape_2d

    grid_2d = Grid(lower, upper, shape)
    get_index_2d = build_indexer(grid_2d)

    # Indexing 2D
    x_lo_up = np.array([-pi, 1])
    x_up_lo_out = np.array([pi, -2])
    assert get_index_2d(x_lo_up) == (UInt32(0), UInt32(32))
    assert get_index_2d(x_up_lo_out) == (UInt32(64), UInt32(32))
