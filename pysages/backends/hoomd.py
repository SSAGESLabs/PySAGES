# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md


import jax

from hoomd.dlext import (
    AccessLocation, AccessMode, HalfStepHook, SystemView,
    net_forces, positions_types, tags, velocities_masses,
)

from jax.dlpack import from_dlpack as wrap


if hasattr(AccessLocation, 'OnDevice'):
    DEFAULT_DEVICE = AccessLocation.OnDevice
else:
    DEFAULT_DEVICE = AccessLocation.OnHost


def is_on_gpu(context):
    return context.on_gpu()


def view(context):
    #
    dt = context.integrator.dt
    system_view = SystemView(context.system_definition)
    #
    positions = wrap(positions_types(system_view, DEFAULT_DEVICE, AccessMode.ReadWrite))
    momenta = wrap(velocities_masses(system_view, DEFAULT_DEVICE, AccessMode.ReadWrite))
    forces = wrap(net_forces(system_view, DEFAULT_DEVICE, AccessMode.ReadWrite))
    ids = wrap(tags(system_view, DEFAULT_DEVICE, AccessMode.ReadWrite))
    #
    box = system_view.particle_data().getGlobalBox()
    L  = box.getL()
    xy = box.getTiltFactorXY()
    xz = box.getTiltFactorXZ()
    yz = box.getTiltFactorYZ()
    lo = box.getLo()
    H = (
        (L.x, xy * L.y, xz * L.z, 0.0),  # Last column is a hack until
        (0.0,      L.y, yz * L.z, 0.0),  # https://github.com/google/jax/issues/4196
        (0.0,      0.0,      L.z, 0.0)   # gets fixed
    )
    origin = (lo.x, lo.y, lo.z)
    #
    return (positions, momenta, forces, ids, H, origin, dt)


class Hook(HalfStepHook):
    def initialize_from(self, sampler, bias):
        snapshot, initialize, update = sampler
        self.snapshot = snapshot
        self.state = initialize()
        self.update_from = update
        self.bias = bias
        return None
    #
    def update(self, timestep):
        self.state = self.update_from(self.snapshot, self.state)
        self.bias(self.snapshot, self.state)
        return None


def attach(context, hook):
    context.integrator.cpp_integrator.setHalfStepHook(hook)
    return None
