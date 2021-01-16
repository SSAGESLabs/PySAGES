# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import jax.numpy as np

from collections import namedtuple
from pysages.utils import register_pytree_namedtuple


@register_pytree_namedtuple
class Optimizer(namedtuple("Optimizer", ("init", "condition", "update"))):
    pass


@register_pytree_namedtuple
class LevenbergMaquardtBayes(
    namedtuple(
        "LevenbergMaquardtBayes",
        ("μi", "μs", "μmin", "μmax"),
        defaults=(
            np.float32(0.005),  # μi
            np.float32(10),     # μs
            np.float32(5e-16),  # μmin
            np.float32(1e10)    # μmax
        )
    )
):
    pass
