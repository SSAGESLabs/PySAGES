# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
This module defines the Sampler class, which is a LAMMPS fix that enables any PySAGES
SamplingMethod to be hooked to a LAMMPS simulation instance.
"""

import importlib
import weakref
from functools import partial

import jax
from jax import jit
from jax import numpy as np
from jax import vmap
from jax.dlpack import from_dlpack
from lammps import dlext
from lammps.dlext import ExecutionSpace, FixDLExt, LAMMPSView, has_kokkos_cuda_enabled

from pysages.backends import snapshot as pbs
from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.typing import Callable, Optional
from pysages.utils import copy, identity

kConversionFactor = {"real": 2390.0573615334906, "metal": 1.0364269e-4, "electron": 1.06657236}
kDefaultLocation = dlext.kOnHost if not hasattr(ExecutionSpace, "kOnDevice") else dlext.kOnDevice


class Sampler(FixDLExt):  # pylint: disable=R0902
    """
    LAMMPS fix that connects PySAGES sampling methods to LAMMPS simulations.

    Parameters
    ----------
    context: ``lammps.core.lammps``
        The LAMMPS simulation instance to which the PySAGES sampling
        machinery will be hooked.
    sampling_method: ``SamplingMethod``
        The sampling method used.
    callbacks: ``Optional[Callback]``
        An optional callback. Some methods define one for logging,
        but it can also be user-defined.
    location: ``lammps.dlext.ExecutionSpace``
        Device where the simulation data will be retrieved.
    """

    def __init__(
        self, context, sampling_method, callback: Optional[Callable], location=kDefaultLocation
    ):
        super().__init__(context)

        on_gpu = (location != dlext.kOnHost) & has_kokkos_cuda_enabled(context)
        location = location if on_gpu else dlext.kOnHost

        self.context = context
        self.location = location
        self.view = LAMMPSView(context)

        helpers, restore, bias = build_helpers(context, sampling_method, on_gpu, pbs.restore)
        initial_snapshot = self.take_snapshot()
        _, initialize, method_update = sampling_method.build(initial_snapshot, helpers)

        self.callback = callback
        self.snapshot = initial_snapshot
        self.state = initialize()
        self._restore = restore
        self._update_box = lambda: self.snapshot.box

        def update(timestep):
            self.view.synchronize()
            self.snapshot = self._update_snapshot()
            self.state = method_update(self.snapshot, self.state)
            bias(self.snapshot, self.state)
            if self.callback:
                self.callback(self.snapshot, self.state, timestep)

        self.set_callback(update)

    def _partial_snapshot(self, include_masses: bool = False):
        positions = from_dlpack(dlext.positions(self.view, self.location))
        types = from_dlpack(dlext.types(self.view, self.location))
        velocities = from_dlpack(dlext.velocities(self.view, self.location))
        forces = from_dlpack(dlext.forces(self.view, self.location))
        tags_map = from_dlpack(dlext.tags_map(self.view, self.location))
        imgs = from_dlpack(dlext.images(self.view, self.location))

        masses = None
        if include_masses:
            masses = from_dlpack(dlext.masses(self.view, self.location))
        vel_mass = (velocities, (masses, types))

        return Snapshot(positions, vel_mass, forces, tags_map, imgs, None, None)

    def _update_snapshot(self):
        s = self._partial_snapshot()
        velocities, (_, types) = s.vel_mass
        _, (masses, _) = self.snapshot.vel_mass
        vel_mass = (velocities, (masses, types))
        box = self._update_box()
        dt = self.snapshot.dt

        return Snapshot(s.positions, vel_mass, s.forces, s.ids[1:], s.images, box, dt)

    def restore(self, prev_snapshot):
        """Replaces this sampler's snapshot with `prev_snapshot`."""
        self._restore(self.snapshot, prev_snapshot)

    def take_snapshot(self):
        """Returns a copy of the current snapshot of the system."""
        s = self._partial_snapshot(include_masses=True)
        box = Box(*get_global_box(self.context))
        dt = get_timestep(self.context)

        return Snapshot(
            copy(s.positions), copy(s.vel_mass), copy(s.forces), s.ids[1:], copy(s.images), box, dt
        )


def build_helpers(context, sampling_method, on_gpu, restore_fn):
    """
    Builds helper methods used for restoring snapshots and biasing a simulation.
    """
    utils = importlib.import_module(".utils", package="pysages.backends")
    dim = context.extract_setting("dimension")
    units = context.extract_global("units")
    factor = kConversionFactor.get(units)

    to_force_units = identity if factor is None else (lambda x: factor * x)

    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if on_gpu:
        sync_forces, view = utils.cupy_helpers()

    else:
        view = utils.view

        def sync_forces():
            pass

    def restore_vm(view, snapshot, prev_snapshot):
        velocities = view(snapshot.vel_mass[0])
        masses_types = snapshot.vel_mass[1]
        masses = view(masses_types[0])
        types = view(masses_types[1])
        prev_masses_types = prev_snapshot.vel_mass[1]
        velocities[:] = view(prev_snapshot.vel_mass[0])
        masses[:] = view(prev_masses_types[0])
        types[:] = view(prev_masses_types[1])

    # TODO: check if this can be sped up.  # pylint: disable=W0511
    def bias(snapshot, state):
        """Adds the computed bias to the forces."""
        if state.bias is None:
            return
        forces = view(snapshot.forces)
        biases = view(state.bias.block_until_ready())
        forces[:, :3] += biases
        sync_forces()

    snapshot_methods = build_snapshot_methods(sampling_method, on_gpu)
    flags = sampling_method.snapshot_flags
    restore = partial(restore_fn, view, restore_vm=restore_vm)
    helpers = HelperMethods(
        build_data_querier(snapshot_methods, flags), lambda: dim, to_force_units
    )

    return helpers, restore, bias


def build_snapshot_methods(sampling_method, on_gpu):
    """
    Builds methods for retrieving snapshot properties in a format useful for collective
    variable calculations.
    """

    if sampling_method.requires_box_unwrapping:
        device = jax.devices("gpu" if on_gpu else "cpu")[0]
        dtype = np.int64 if dlext.kImgBitSize == 64 else np.int32
        offset = dlext.kImgMax

        with jax.default_device(device):
            bits = np.asarray((0, dlext.kImgBits, dlext.kImg2Bits), dtype=dtype)
            mask = np.asarray((dlext.kImgMask, dlext.kImgMask, -1), dtype=dtype)

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
        M = masses[types].reshape(-1, 1)
        return (M * V).flatten()

    @jit
    def masses(snapshot):
        return snapshot.vel_mass[:, 3:]

    return SnapshotMethods(jit(positions), indices, momenta, masses)


def get_global_box(context):
    """Get the box and origin of a LAMMPS simulation."""
    boxlo, boxhi, xy, yz, xz, *_ = context.extract_box()
    Lx = boxhi[0] - boxlo[0]
    Ly = boxhi[1] - boxlo[1]
    Lz = boxhi[2] - boxlo[2]
    origin = boxlo
    H = ((Lx, xy * Ly, xz * Lz), (0.0, Ly, yz * Lz), (0.0, 0.0, Lz))
    return H, origin


def get_timestep(context):
    """Get the timestep of a LAMMPS simulation."""
    return context.extract_global("dt")


def bind(sampling_context: SamplingContext, callback: Optional[Callable], **kwargs):
    """
    Defines and sets up a Sampler to perform an enhanced sampling simulation.

    This function takes a ``sampling_context`` that has its context attribute as an instance
    of a LAMMPS simulation, and creates a ``Sampler`` object that connects the PySAGES
    sampling method to the LAMMPS simulation. It also modifies the sampling_context's view
    and run attributes to use the sampler's view and the LAMMPS run command.
    """
    identity(kwargs)  # we ignore the kwargs for now

    context = sampling_context.context
    sampling_method = sampling_context.method
    sampler = Sampler(context, sampling_method, callback)
    sampling_context.view = sampler.view
    sampling_context.run = lambda n, **kwargs: context.command(f"run {n}")

    # We want to support backends that also are context managers as long
    # as the simulation is kept alive after exiting the context.
    # Unfortunately, the default implementation of `lammps.__exit__` closes
    # the lammps instance, so we need to overwrite it.
    context.__exit__ = lambda *args: None
    # Ensure that the lammps context is properly finalized.
    weakref.finalize(context, context.finalize)

    return sampler
