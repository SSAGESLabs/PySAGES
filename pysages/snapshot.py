# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


from collections import namedtuple
from .utils import register_pytree_namedtuple


BoxInfo = namedtuple("BoxInfo", ["H", "origin"])

SnapshotData = namedtuple(
    "Snapshot",
    ["positions", "vel_mass", "forces", "tags", "box", "dt"]
)


@register_pytree_namedtuple
class Box(BoxInfo):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


@register_pytree_namedtuple
class SystemView(SnapshotData):
    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)
