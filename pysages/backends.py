# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta (see LICENSE.md)


import importlib


# Records the backend selected with `set_backend`
_ACTIVE_BACKEND = None


def active_backend():
    return _ACTIVE_BACKEND


def supported_backends():
    return ('hoomd',)


def set_backend(name):
    """To see a list of possible backends run `supported_backends()`."""
    #
    global _ACTIVE_BACKEND
    #
    if name in supported_backends():
        _ACTIVE_BACKEND = importlib.import_module(name + '.dlext')
    else:
        raise ValueError('Invalid backend')
    #
    return _ACTIVE_BACKEND