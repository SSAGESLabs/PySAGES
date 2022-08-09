#!/usr/bin/env python3

import sys
import numpy as np
import gsd
import gsd.hoomd


class System:
    def __init__(self):
        self.L = 5
        self.N = 200


def post_process_pos(snapshot):
    box_size = snapshot.configuration.box[:3]
    snapshot.particles.image = np.rint(snapshot.particles.position / box_size)
    snapshot.particles.position -= snapshot.particles.image * box_size
    return snapshot


def get_snap(system):
    L = system.L
    snapshot = gsd.hoomd.Snapshot()
    snapshot.configuration.box = [L, L, L, 0, 0, 0]

    snapshot.particles.N = N = system.N

    snapshot.particles.types = ["A", "B"]
    snapshot.particles.position = np.zeros((N, 3))
    snapshot.particles.velocity = np.random.standard_normal((N, 3))
    snapshot.particles.image = np.zeros((N, 3), dtype=int)
    snapshot.particles.typeid = np.zeros(N, dtype=int)
    snapshot.particles.typeid += 1
    snapshot.particles.typeid[0] = 0

    rng = np.random.default_rng()
    for particle in range(N):
        snapshot.particles.position[particle, 0] = rng.random() * L - L / 2
        snapshot.particles.position[particle, 1] = rng.random() * L - L / 2
        snapshot.particles.position[particle, 2] = rng.random() * L - L / 2

    snapshot.particles.position[0, 0] = -np.pi
    snapshot.particles.position[0, 1] = -np.pi
    snapshot.particles.position[0, 1] = -np.pi

    return snapshot


def main(argv):
    if len(argv) != 1:
        print("Usage: ./gen_gsd.py")

    system = System()
    snap = get_snap(system)
    snap = post_process_pos(snap)
    snap.particles.validate()
    with gsd.hoomd.open("start.gsd", "wb") as f:
        f.append(snap)


if __name__ == "__main__":
    main(sys.argv)
