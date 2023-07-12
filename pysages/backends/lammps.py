# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

# Maintainer: ndtrung

import importlib
from functools import partial

import jax
from jax import jit, vmap
from jax import numpy as np
from jax.dlpack import from_dlpack as asarray
from lammps import dlext
from lammps.dlext import (
    ExecutionSpace,
    FixDLExt,
    kImgBitSize,
    kImgBits,
    kImg2Bits,
    kImgMask,
    kImgMax,
)

from pysages.backends import snapshot as ps
from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.typing import Callable, Optional
from pysages.utils import copy

kDefaultLocation = (
    ExecutionSpace.kOnHost if not hasattr(ExecutionSpace, "kOnDevice") else ExecutionSpace.kOnDevice
)


class Sampler(FixDLExt):
    def __init__(
        self, context, sampling_method, callback: Optional[Callable], location=kDefaultLocation
    ):
        super().__init__(context, "1 all dlext space device".split())

        on_gpu = (location != dlext.kOnHost) & self.view.has_kokkos_cuda_enabled()
        helpers, restore, bias = build_helpers(context, sampling_method, on_gpu, ps.restore)

        initial_snapshot = take_snapshot(context, self.view, location=location)
        _, initialize, method_update = sampling_method.build(initial_snapshot, helpers)

        self.location = location
        self.state = initialize()
        self.bias = partial(bias, sync_backend=self.view.synchronize)
        self.callback = callback
        self.snapshot = initial_snapshot
        self.restore = lambda prev_snapshot: restore(self.snapshot, prev_snapshot)
        self.take_snapshot = partial(take_snapshot, context, self.view)

        def update_box():
            return self.snapshot.box

        def update_snapshot():
            view = self.view
            location = self.location

            positions = asarray(dlext.positions(view, location))
            types = asarray(dlext.types(view, location))
            velocities = asarray(dlext.velocities(view, location))
            forces = asarray(dlext.forces(view, location))
            tags_map = asarray(dlext.tags_map(view, location))[1:]
            imgs = asarray(dlext.images(view, location))

            _, (masses, _) = self.snapshot.vel_mass
            vel_mass = (velocities, (masses, types))
            box = update_box()
            dt = self.snapshot.dt

            return Snapshot(positions, vel_mass, forces, tags_map, imgs, box, dt)

        def update(timestep):
            self.snapshot = update_snapshot()
            self.state = method_update(self.snapshot, self.state)
            self.bias(self.snapshot, self.state)
            if self.callback:
                self.callback(self.snapshot, self.state, timestep)

        self.update = update
        self.set_callback(self.update)


def build_helpers(context, sampling_method, on_gpu, restore_fn):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if on_gpu:
        cupy = importlib.import_module("cupy")
        view = cupy.asarray

        def sync_forces():
            cupy.cuda.get_current_stream().synchronize()

    else:
        utils = importlib.import_module(".utils", package="pysages.backends")
        view = utils.view

        def sync_forces():
            pass

    def bias(snapshot, state, sync_backend):
        """Adds the computed bias to the forces."""
        # TODO: check if this can be JIT compiled with numba.
        # Forces may be computed asynchronously on the GPU, so we need to
        # synchronize them before applying the bias.
        if state.bias is None:
            return
        sync_backend()
        forces = view(snapshot.forces)
        biases = view(state.bias.block_until_ready())
        forces[:, :3] += biases
        sync_forces()

    snapshot_methods = build_snapshot_methods(sampling_method, on_gpu)
    flags = sampling_method.snapshot_flags
    restore = partial(restore_fn, view)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), get_dimension(context))

    return helpers, restore, bias


def build_snapshot_methods(sampling_method, on_gpu):
    if sampling_method.requires_box_unwrapping:
        device = jax.devices("gpu" if on_gpu else "cpu")[0]
        dtype = np.int64 if kImgBitSize == 64 else np.int32
        all_mask = 2 ** (kImgBitSize - 1) - 1
        offset = kImgMax

        with jax.default_device(device):
            bits = np.asarray((0, kImgBits, kImg2Bits), dtype=dtype)
            mask = np.asarray((kImgMask, kImgMask, all_mask), dtype=dtype)

        def unpack(image):
            return (image >> bits & mask) - offset

        def positions(snapshot):
            L = np.diag(snapshot.box.H)
            return snapshot.positions[:, :3] + L * vmap(unpack)(snapshot.images)

    else:

        def positions(snapshot):
            return snapshot.positions

    @jit
    def indices(snapshot):
        return snapshot.ids

    @jit
    def momenta(snapshot):
        V, (masses, types) = snapshot.vel_mass
        M = masses[types]
        return (M * V).flatten()

    @jit
    def masses(snapshot):
        return snapshot.vel_mass[:, 3:]

    return SnapshotMethods(jit(positions), indices, momenta, masses)


def get_dimension(context):
    return context.extract_setting("dimension")


def get_global_box(context):
    boxlo, boxhi, xy, yz, xz, *_ = context.extract_box()
    Lx = boxhi[0] - boxlo[0]
    Ly = boxhi[1] - boxlo[1]
    Lz = boxhi[2] - boxlo[2]
    origin = boxlo
    H = ((Lx, xy * Ly, xz * Lz), (0.0, Ly, yz * Lz), (0.0, 0.0, Lz))
    return H, origin


def get_timestep(context):
    return context.extract_global("dt")


def take_snapshot(context, view, location=kDefaultLocation):
    positions = copy(asarray(dlext.positions(view, location)))
    types = copy(asarray(dlext.types(view, location)))
    velocities = copy(asarray(dlext.velocities(view, location)))
    masses = copy(asarray(dlext.masses(view, location)))
    forces = copy(asarray(dlext.forces(view, location)))
    tags_map = asarray(dlext.tags_map(view, location))[1:]
    imgs = copy(asarray(dlext.images(view, location)))

    vel_mass = (velocities, (masses, types))

    box = Box(*get_global_box(context))
    dt = get_timestep(context)

    return Snapshot(positions, vel_mass, forces, tags_map, imgs, box, dt)


def bind(sampling_context: SamplingContext, callback: Optional[Callable], **kwargs):
    context = sampling_context.context
    sampling_method = sampling_context.method
    sampler = Sampler(context, sampling_method, callback)
    sampling_context.view = sampler.view
    sampling_context.run = lambda n, **kwargs: context.command(f"run {n}")
    return sampler
