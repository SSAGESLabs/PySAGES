# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from typing import Callable, NamedTuple

from jax import jit
from jax import numpy as np
from jax_md import dataclasses

from pysages.backends.core import ContextWrapper
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.methods import SamplingMethod
from pysages.utils import copy


class Sampler:
    def __init__(self, method_bundle, context_state):
        snapshot, initialize, update = method_bundle
        self.context_state = context_state
        self.snapshot = snapshot
        self.state = initialize()
        self.update = update

    def restore(self, prev_snapshot):
        self.snapshot = prev_snapshot

    def take_snapshot(self):
        return copy(self.snapshot)


def take_snapshot(state, box, dt):
    dims = box.shape[0]
    positions = state.position
    forces = getattr(state, "force", state.position)
    ids = np.arange(len(positions))
    velocities = getattr(state, "velocity", state.position)
    masses = state.mass.reshape(-1, 1)
    vel_mass = (velocities, masses)
    origin = tuple(0.0 for _ in range(dims))
    return Snapshot(positions, vel_mass, forces, ids, None, Box(box, origin), dt)


def update_snapshot(snapshot, state):
    _, masses = snapshot.vel_mass
    positions = state.position
    vel_mass = (state.velocity, masses)
    forces = state.force
    return snapshot._replace(positions=positions, vel_mass=vel_mass, forces=forces)


def build_snapshot_methods(context, sampling_method):
    def indices(snapshot):
        return snapshot.ids

    def masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    def positions(snapshot):
        return snapshot.positions

    def momenta(snapshot):
        V, M = snapshot.vel_mass
        return (V * M).flatten()

    return SnapshotMethods(positions, indices, jit(momenta), masses)


def build_helpers(context, sampling_method):
    def dimensionality():
        return context.box.shape[0]

    snapshot_methods = build_snapshot_methods(context, sampling_method)
    flags = sampling_method.snapshot_flags
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), dimensionality)

    return helpers


def build_runner(context, sampler, callback, jit_compile=True):
    step_fn = context.step_fn

    def _step(wrapped_context_state, snapshot, sampler_state):
        context_state = wrapped_context_state.state
        snapshot = update_snapshot(snapshot, context_state)
        wrapped_context_state = step_fn(wrapped_context_state)  # jax_md simulation step
        sampler_state = sampler.update(snapshot, sampler_state)  # pysages update
        if sampler_state.bias is not None:  # bias the simulation
            context_state = wrapped_context_state.state
            biased_forces = context_state.force + sampler_state.bias
            context_state = dataclasses.replace(context_state, force=biased_forces)
            wrapped_context_state = wrapped_context_state._replace(state=context_state)
        return wrapped_context_state, snapshot, sampler_state

    step = jit(_step) if jit_compile else _step

    def run(timesteps):
        # TODO: Allow to optionally batch timesteps with `lax.fori_loop`
        for i in range(timesteps):
            context_state, snapshot, state = step(
                sampler.context_state, sampler.snapshot, sampler.state
            )
            sampler.context_state = context_state
            sampler.snapshot = snapshot
            sampler.state = state
            if callback:
                callback(sampler.snapshot, sampler.state, i)

    return run


class View(NamedTuple):
    synchronize: Callable


def bind(
    wrapped_context: ContextWrapper, sampling_method: SamplingMethod, callback: Callable, **kwargs
):
    context = wrapped_context.context
    context_state = context.init_fn(**kwargs)
    snapshot = take_snapshot(context_state.state, context.box, context.dt)
    helpers = build_helpers(context, sampling_method)
    method_bundle = sampling_method.build(snapshot, helpers)
    sampler = Sampler(method_bundle, context_state)
    wrapped_context.view = View((lambda: None))
    wrapped_context.run = build_runner(
        context, sampler, callback, jit_compile=kwargs.get("jit_compile", True)
    )
    return sampler
