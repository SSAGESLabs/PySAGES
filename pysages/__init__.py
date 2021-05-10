# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)

from ._version import get_versions

from .backends import (
    bind,
    set_backend,
    supported_backends,
)

from .ssages import (
    Grid,
    cvs,
    methods,
)

__version__ = get_versions()['version']
del get_versions
