# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


from collections import namedtuple
from pysages.utils import register_pytree_namedtuple


@register_pytree_namedtuple
class Box(namedtuple("BoxInfo", ("H", "origin"))):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


@register_pytree_namedtuple
class SystemView(
    namedtuple(
        "Snapshot",
        ("positions", "vel_mass", "forces", "tags", "box", "dt")
    )
):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)