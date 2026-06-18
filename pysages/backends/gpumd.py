# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES
#
# Author: Jaafar Mehrez
# (Shanghai Jiao Tong University, Shanghai, China;
#  HPQC Labs, Waterloo, Canada;
#  jaafarmehrez@sjtu.edu.cn, jaafar@hpqc.org)

"""
PySAGES backend for GPUMD.

This backend assumes that GPUMD has been compiled with a pybind11 Python wrapper
that exposes the simulation data as DLPack capsules and supports a per-step
Python callback.

The wrapper module is expected to be importable as ``gpumd``.
"""

import time
from functools import partial

from jax import jit
from jax import numpy as np
from jax.dlpack import from_dlpack

from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.backends.snapshot import restore as _restore
from pysages.typing import Callable, Optional
from pysages.utils import copy, identity

# The gpumd package does not exist yet; importing it here will raise a clear
# error when a user tries to use the backend without installing the wrapper.
try:
    import gpumd
except ImportError as err:
    raise ImportError(
        "The gpumd Python package is required for the GPUMD backend. "
        "Please build and install the GPUMD pybind11 wrapper."
    ) from err


class Sampler:
    """
    GPUMD sampler that connects a PySAGES sampling method to a GPUMD simulation.

    Parameters
    ----------
    simulation: gpumd.Simulation
        The wrapped GPUMD simulation instance.

    sampling_method: pysages.methods.SamplingMethod
        The enhanced-sampling method to use.

    callback: Optional[Callable]
        Optional user callback for logging or analysis.
    """

    def __init__(
        self,
        simulation,
        sampling_method,
        callback: Optional[Callable],
    ):
        self.simulation = simulation
        self.callback = callback

        # Build initial snapshot and sampling method
        initial_snapshot = self._take_snapshot()
        helpers, restore, bias = build_helpers(simulation, sampling_method)
        _, initialize, method_update = sampling_method.build(initial_snapshot, helpers)

        self.snapshot = initial_snapshot
        self.state = initialize()
        self._restore = restore
        self._bias = bias
        self._method_update = method_update

        self._cached_masses = initial_snapshot.vel_mass[1]
        self._cached_ids = initial_snapshot.ids
        self._cached_box = initial_snapshot.box
        self._cached_dt = initial_snapshot.dt

        self._box_is_constant = simulation.is_box_constant()

        self._timings = {
            "snapshot": 0.0,
            "update": 0.0,
            "bias": 0.0,
            "callback": 0.0,
            "total": 0.0,
        }
        self._timing_count = 0

        simulation.clear_external_bias()
        simulation.set_step_callback(self._update)

    def _update(self, timestep: int):
        """
        Callback executed by GPUMD every timestep.
        """
        t0 = time.perf_counter()

        self.snapshot = self._build_snapshot_with_fresh_arrays(timestep)
        t1 = time.perf_counter()

        self.state = self._method_update(self.snapshot, self.state)
        t2 = time.perf_counter()
        self._bias(self.snapshot, self.state)
        t3 = time.perf_counter()
        if self.callback:
            self.callback(self.snapshot, self.state, timestep)
        t4 = time.perf_counter()

        self._timings["snapshot"] += t1 - t0
        self._timings["update"] += t2 - t1
        self._timings["bias"] += t3 - t2
        self._timings["callback"] += t4 - t3
        self._timings["total"] += t4 - t0
        self._timing_count += 1

    def restore(self, prev_snapshot):
        self._restore(self.snapshot, prev_snapshot)

    def take_snapshot(self):
        return copy(self.snapshot)

    def print_timings(self):
        n = self._timing_count
        if n == 0:
            print("[gpumd backend] No timing data collected yet.")
            return
        print("[gpumd backend] Per-step timing breakdown (ms):")
        print(f"  {'Stage':<12} {'Total (ms)':<14} {'Per-step (ms)':<14} {'% of total':<10}")
        print("  " + "-" * 52)
        total_ms = self._timings["total"] * 1000.0
        for key in ["snapshot", "update", "bias", "callback"]:
            stage_ms = self._timings[key] * 1000.0
            per_ms = stage_ms / n
            pct = (stage_ms / total_ms * 100.0) if total_ms > 0 else 0.0
            print(f"  {key:<12} {stage_ms:>10.2f}     {per_ms:>10.4f}     {pct:>6.2f}")
        print("  " + "-" * 52)
        print(f"  {'total':<12} {total_ms:>10.2f}     {total_ms / n:>10.4f}     {'100.00':>6}")
        print(f"  Steps counted: {n}")

    def _build_snapshot_with_fresh_arrays(self, timestep: int):
        """
        Rebuild snapshot with fresh DLPack for positions/velocities/forces.

        Constant data (ids, dt, masses) is reused from the cache built during
        ``__init__``.  The simulation box is also cached, but for NPT or
        change_box runs it may vary.  We query ``sim.is_box_constant()`` once
        during ``__init__``; if the box is constant we never call ``get_box()``
        again, otherwise we refresh it every 100 steps.
        """
        sim = self.simulation
        positions = from_dlpack(sim.get_positions_dlpack()).T
        velocities = from_dlpack(sim.get_velocities_dlpack()).T
        forces = from_dlpack(sim.get_forces_dlpack()).T
        vel_mass = (velocities, self._cached_masses)

        if not self._box_is_constant and timestep % 100 == 0:
            h, origin = sim.get_box()
            cached_h = self._cached_box.H
            h_3x3 = np.asarray(h).reshape(3, 3)
            if not np.allclose(cached_h, h_3x3, atol=1e-12):
                H = (
                    (h[0], h[1], h[2]),
                    (h[3], h[4], h[5]),
                    (h[6], h[7], h[8]),
                )
                self._cached_box = Box(H, origin)

        return Snapshot(
            positions,
            vel_mass,
            forces,
            self._cached_ids,
            self._cached_box,
            self._cached_dt,
        )

    def _take_snapshot(self):
        """
        Construct a full PySAGES Snapshot from the current GPUMD state.

        Called once during ``Sampler.__init__`` to build the initial snapshot
        and populate the constant-data cache.  During normal MD
        ``_build_snapshot_with_fresh_arrays`` is used instead so that JAX
        sees updated GPU values each step.
        """
        sim = self.simulation
        positions = from_dlpack(sim.get_positions_dlpack()).T
        velocities = from_dlpack(sim.get_velocities_dlpack()).T
        forces = from_dlpack(sim.get_forces_dlpack()).T
        masses = from_dlpack(sim.get_masses_dlpack())
        types = from_dlpack(sim.get_types_dlpack())
        vel_mass = (velocities, masses)
        ids = np.arange(types.size)
        h, origin = sim.get_box()
        H = (
            (h[0], h[1], h[2]),
            (h[3], h[4], h[5]),
            (h[6], h[7], h[8]),
        )
        box = Box(H, origin)
        dt = sim.get_timestep()

        return Snapshot(positions, vel_mass, forces, ids, box, dt)


def build_snapshot_methods(context, sampling_method):
    """
    Build methods for retrieving snapshot properties in a format useful for
    collective variable calculations.
    """

    if sampling_method.requires_box_unwrapping:

        def positions(snapshot):
            pos = snapshot.positions[:, :3]
            L = np.diag(snapshot.box.H)
            ref = pos[0]
            delta = pos - ref
            images = np.rint(delta / L)
            return pos - L * images

    else:

        def positions(snapshot):
            return snapshot.positions

    @jit
    def indices(snapshot):
        return snapshot.ids

    @jit
    def momenta(snapshot):
        velocities, masses = snapshot.vel_mass
        return (masses * velocities).flatten()

    @jit
    def masses(snapshot):
        return snapshot.vel_mass[1]

    return SnapshotMethods(jit(positions), indices, momenta, masses)


def build_helpers(context, sampling_method):
    """
    Builds helper methods used for restoring snapshots and biasing a simulation.
    """
    utils = __import__("pysages.backends.utils", fromlist=["cupy_helpers"])

    sync_forces, view = utils.cupy_helpers()

    def restore_vm(view, snapshot, prev_snapshot):

        velocities = view(snapshot.vel_mass[0])
        masses = view(snapshot.vel_mass[1])
        prev_velocities = view(prev_snapshot.vel_mass[0])
        prev_masses = view(prev_snapshot.vel_mass[1])
        velocities[:] = prev_velocities
        masses[:] = prev_masses

    import jax.dlpack

    def bias(snapshot, state):
        """
        Adds the computed bias to GPUMD's force_per_atom via a custom CUDA kernel.

        We intentionally skip ``block_until_ready()`` here because the
        C++ side calls ``cudaDeviceSynchronize()`` after launching the kernel,
        which waits for all GPU streams (including JAX's) to finish.
        """
        if state.bias is None:
            return

        bias_arr = state.bias
        if bias_arr.dtype != np.float64:
            bias_arr = bias_arr.astype(np.float64)

        context.add_aos_bias_to_forces(jax.dlpack.to_dlpack(bias_arr.flatten()))

    def dimensionality():
        return 3

    snapshot_methods = build_snapshot_methods(context, sampling_method)
    flags = sampling_method.snapshot_flags
    restore = partial(_restore, view, restore_vm=restore_vm)
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), dimensionality)

    return helpers, restore, bias


def bind(sampling_context: SamplingContext, callback: Optional[Callable] = None, **kwargs):
    """
    Bind a PySAGES sampling method to a GPUMD simulation.

    Parameters
    ----------
    sampling_context: pysages.backends.core.SamplingContext
        The PySAGES sampling context, whose ``.context`` attribute is a
        ``gpumd.Simulation`` instance.

    callback: Optional[Callable]
        User callback executed after the bias has been applied each step.

    Returns
    -------
    Sampler
        The sampler object managing the GPUMD / PySAGES integration.
    """
    identity(kwargs)  # reserved for future options

    simulation = sampling_context.context
    sampling_method = sampling_context.method
    sampler = Sampler(simulation, sampling_method, callback)
    sampling_context.run = simulation.run

    return sampler
