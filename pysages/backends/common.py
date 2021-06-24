# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES


def restore(view, restore_vm, snapshot, prev_snapshot):
    # Create a mutable view of the jax arrays
    positions = view(snapshot.positions)
    forces = view(snapshot.forces)
    ids = view(snapshot.ids)
    # Overwrite the data
    positions[:] = view(prev_snapshot.positions)
    forces[:] = view(prev_snapshot.forces)
    ids[:] = view(prev_snapshot.ids)
    # Special handling for velocities and masses
    restore_vm(snapshot, prev_snapshot)


# Fallback method for restoring velocities and masses
def restore_vm(view, snapshot, prev_snapshot):
    vel_mass = view(snapshot.vel_mass)
    vel_mass[:] = view(prev_snapshot.vel_mass)
