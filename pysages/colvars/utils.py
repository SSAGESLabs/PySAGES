# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Extra utilities for computations involving collective variables.
"""

from jax import numpy as np

from pysages.colvars.angles import Angle, DihedralAngle


def get_periods(cvs):
    """
    Returns an array with `2 * np.pi` for each cv in `cvs` that is of periodic type
    and `inf` for the rest of the entries.
    """
    periodic_types = (Angle, DihedralAngle)
    return np.array([2 * np.pi if type(cv) in periodic_types else np.inf for cv in cvs])


def wrap(x, P):
    """
    Given a period `P`, wraps around `x` over the interval from `-P / 2` to `P / 2`.
    """
    return np.where(np.isinf(P), x, x - (np.abs(x) > P / 2) * np.sign(x) * P)


def rational_switching_function(r, r_0, d_0=0.0, n=6, m=None):
    """
    Rational switching function applied to a given variable r.
    
    Parameters
    ----------
    r: float
        variable to which switching function is applied.
    
    r_0 : float
        
    d_0: float = 0.0
    
    n: int = 6
    
    m: int = 2*n

    Returns
    -------
    s : float
        Rational switching function applied to a given r.
    """
    
    if m == None:
        m = 2*n
    
    s_common = (r - d_0)/r_0
    s_n = 1 - s_common**n
    s_m = 1 - s_common**m
    s = s_n/s_m
         
    return s
    