# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jax import jit
from jax import numpy as np
from jax_md import dataclasses

from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.typing import Callable, NamedTuple
from pysages.utils import check_device_array, copy
import jax.lax

class Sampler:
    def __init__(self, method_bundle, context_state, callback: Callable):
        initial_snapshot, initialize, method_update = method_bundle
        self.state = initialize()
        self.callback = callback
        self.context_state = context_state
        self.snapshot = initial_snapshot
        self.update = method_update

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

    if state.chain:
        chain_data = vars(state.chain)
    else:
        chain_data = None

    check_device_array(positions)  # currently, we only support `DeviceArray`s

    return Snapshot(positions, vel_mass, forces, ids, None, Box(box, origin), dt, chain_data=chain_data)


def update_snapshot(snapshot, state):
    _, masses = snapshot.vel_mass
    positions = state.position
    vel_mass = (state.velocity, masses)
    forces = state.force
    if state.chain:
        chain_data = vars(state.chain)
        return snapshot._replace(
                positions=positions,
                vel_mass=vel_mass,
                forces=forces,
                chain_data=chain_data)
    else:
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

jax_fn_container = {'is_defined': False, 'run_fn': None}

def build_runner(context, sampler, jit_compile=True):
    step_fn = context.step_fn

    if not jax_fn_container['is_defined']:
        jax_fn_container['is_defined'] = True

        def _step(sampling_context_state, snapshot, sampler_state):
            sampling_context_state = step_fn(sampling_context_state)  # jax_md simulation step
            context_state = sampling_context_state.state
            snapshot = update_snapshot(snapshot, context_state)
            sampler_state = sampler.update(snapshot, sampler_state)  # pysages update
            if sampler_state.bias is not None:  # bias the simulation
                context_state = sampling_context_state.state
                biased_forces = context_state.force + sampler_state.bias
                context_state = dataclasses.replace(context_state, force=biased_forces)
                sampling_context_state = sampling_context_state._replace(state=context_state)
            return sampling_context_state, snapshot, sampler_state

        step = jit(_step) if jit_compile else _step



        def _run_body(i, input_states_and_snapshots):
            context_state, snapshot, sampler_state = input_states_and_snapshots
            return tuple(step(context_state, snapshot, sampler_state))

        run_body = jit(_run_body) if jit_compile else _run_body
    

        jax_fn_container['run_fn'] = run_body

    def run(timesteps):
        # TODO: Allow to optionally batch timesteps with `lax.fori_loop`

        sampler.context_state, sampler.snapshot, sampler.state = jax.block_until_ready( 
                jax.lax.fori_loop(0, timesteps, jax_fn_container['run_fn'], (sampler.context_state, sampler.snapshot, sampler.state))
            )

    #def run(timesteps):
    #    # TODO: Allow to optionally batch timesteps with `lax.fori_loop`
    #    for i in range(timesteps):
    #        context_state, snapshot, state = step(
    #            sampler.context_state, sampler.snapshot, sampler.state
    #        )
    #        sampler.context_state = context_state
    #        sampler.snapshot = snapshot
    #        sampler.state = state
    #        if sampler.callback:
    #            sampler.callback(sampler.snapshot, sampler.state, i)

    #return run

    return run




class View(NamedTuple):
    synchronize: Callable


def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    context = sampling_context.context
    sampling_method = sampling_context.method
    context_state = context.init_fn(**kwargs)
    snapshot = take_snapshot(context_state.state, context.box, context.dt)
    helpers = build_helpers(context, sampling_method)
    method_bundle = sampling_method.build(snapshot, helpers)
    sampler = Sampler(method_bundle, context_state, callback)
    sampling_context.view = View((lambda: None))
    sampling_context.run = build_runner(
        context, sampler, jit_compile=kwargs.get("jit_compile", True)
    )
    return sampler
