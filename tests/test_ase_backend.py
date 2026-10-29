# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

import numpy as onp
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.md import Langevin, VelocityVerlet
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import pysages
from pysages.colvars import Component
from pysages.methods import HarmonicBias

T, DT = 60.0, 2 * units.fs

INTEGRATORS = {
    "Langevin": lambda a: Langevin(a, DT, temperature_K=T, friction=0.01),
    "NPTBerendsen": lambda a: NPTBerendsen(
        a,
        DT,
        temperature_K=T,
        pressure_au=0.0,
        taut=100 * units.fs,
        taup=1000 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
    ),
    "NPT": lambda a: NPT(
        a,
        DT,
        temperature_K=T,
        externalstress=0.0,
        ttime=25 * units.fs,
        pfactor=(75 * units.fs) ** 2 * 100 * units.GPa,
    ),
}
# Integrators whose two half-kicks per step share a force evaluation
HALF_KICK_INTEGRATORS = {
    "VelocityVerlet": lambda a, dt: VelocityVerlet(a, dt),
    "Langevin": lambda a, dt: Langevin(a, dt, temperature_K=0.0, friction=0.0),
}
try:  # ASE >= 3.24
    from ase.md.nose_hoover_chain import IsotropicMTKNPT

    INTEGRATORS["IsotropicMTKNPT"] = lambda a: IsotropicMTKNPT(
        a, DT, temperature_K=T, pressure_au=0.0, tdamp=100 * units.fs, pdamp=1e3 * units.fs
    )
    HALF_KICK_INTEGRATORS["IsotropicMTKNPT"] = lambda a, dt: IsotropicMTKNPT(
        a, dt, temperature_K=1.0, pressure_au=0.0, tdamp=1e6 * dt, pdamp=1e6 * dt
    )
except ImportError:
    pass


def lattice(temperature=0.0):
    # At rest, interatomic forces cancel by symmetry
    atoms = bulk("Ar", "fcc", a=5.4, cubic=True).repeat((2, 2, 2))
    atoms.calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=8.0)
    if temperature > 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    return atoms


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_one_update_per_step(integrator):
    nsteps, ncalls = 20, []
    method = HarmonicBias([Component([0], 0)], kspring=1e-6, center=0.0)
    make = INTEGRATORS[integrator]
    pysages.run(method, lambda **kw: make(lattice(T)), nsteps, lambda *_: ncalls.append(1))
    # ASE requests the forces several times per step; plus one for the initial configuration
    assert len(ncalls) == nsteps + 1


@pytest.mark.parametrize("integrator", HALF_KICK_INTEGRATORS)
def test_bias_impulse(integrator):
    # A constant force F on atom 0 along x: the harmonic bias centered F / k away from it
    dt, F, k = 0.01 * units.fs, 0.05, 1.0
    atoms = lattice()
    method = HarmonicBias([Component([0], 0)], kspring=k, center=atoms.positions[0, 0] + F / k)
    make = HALF_KICK_INTEGRATORS[integrator]
    pysages.run(method, lambda **kw: make(atoms, dt), 2)
    # Impulse Δp = F t: each step's biased forces enter both of its half-kicks
    assert onp.isclose(atoms.get_momenta()[0, 0], 2 * dt * F, rtol=1e-4)
