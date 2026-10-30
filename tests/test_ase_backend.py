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
from pysages.colvars import BoxVolume, Component, Distance
from pysages.grids import Grid
from pysages.methods import ABF, HarmonicBias, Unbiased

T, DT = 60.0, 2 * units.fs
P_REF = 0.01  # eV/Å^3

INTEGRATORS = {
    "Langevin": lambda a: Langevin(a, DT, temperature_K=T, friction=0.01),
    "NPTBerendsen": lambda a: NPTBerendsen(
        a,
        DT,
        temperature_K=T,
        pressure_au=P_REF,
        taut=100 * units.fs,
        taup=1000 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
    ),
    "NPT": lambda a: NPT(
        a,
        DT,
        temperature_K=T,
        externalstress=P_REF,
        ttime=25 * units.fs,
        pfactor=(75 * units.fs) ** 2 * 100 * units.GPa,
    ),
}
# Integrators whose two half-kicks per step share a force evaluation
HALF_KICK_INTEGRATORS = {
    "VelocityVerlet": lambda a, dt: VelocityVerlet(a, dt),
    "Langevin": lambda a, dt: Langevin(a, dt, temperature_K=0.0, friction=0.0),
}
try:  # ASE >= 3.25
    from ase.md.nose_hoover_chain import IsotropicMTKNPT

    INTEGRATORS["IsotropicMTKNPT"] = lambda a: IsotropicMTKNPT(
        a, DT, temperature_K=T, pressure_au=P_REF, tdamp=100 * units.fs, pdamp=1e3 * units.fs
    )
    HALF_KICK_INTEGRATORS["IsotropicMTKNPT"] = lambda a, dt: IsotropicMTKNPT(
        a, dt, temperature_K=1.0, pressure_au=0.0, tdamp=1e6 * dt, pdamp=1e6 * dt
    )
except ImportError:
    pass
BAROSTATS = [name for name in INTEGRATORS if name != "Langevin"]

VOLUME_GRID = Grid(lower=(1100.0,), upper=(1400.0,), shape=(30,))
BOX_BIASES = {
    "HarmonicBias": lambda: HarmonicBias([BoxVolume()], kspring=1e-3, center=1200.0),
    "ABF": lambda: ABF([BoxVolume()], VOLUME_GRID, N=10),
}
BOX_METHODS = {**BOX_BIASES, "Unbiased": lambda: Unbiased([BoxVolume()])}


def lattice(temperature=0.0):
    # At rest, interatomic forces cancel by symmetry
    atoms = bulk("Ar", "fcc", a=5.4, cubic=True).repeat((2, 2, 2))
    atoms.calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=8.0)
    if temperature > 0:
        rng = onp.random.default_rng(0)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
    return atoms


def reference_stress(atoms):
    """Virial stress of `atoms` from a calculator PySAGES has not wrapped."""
    atoms = atoms.copy()
    atoms.calc = lattice().calc
    return atoms.get_stress(voigt=False)


def sym(M):
    return (M + M.T) / 2


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


@pytest.mark.parametrize("method", BOX_METHODS)
def test_box_cvs_need_a_barostat(method):
    with pytest.raises(ValueError, match="constant-pressure integrator"):
        pysages.run(BOX_METHODS[method](), lambda **kw: INTEGRATORS["Langevin"](lattice(T)), 1)


@pytest.mark.parametrize("integrator", BAROSTATS)
@pytest.mark.parametrize("method", BOX_BIASES)
def test_snapshot_pressure(integrator, method):
    # Under a barostat the snapshot carries P - P_ref·I as the barostat sees it: the calculator's
    # virial, the ideal-gas term, and the box bias applied since the previous update
    nsteps, atoms, prev, ncalls = 10, lattice(T), {"cell_bias": onp.zeros((3, 3))}, []

    def check(snapshot, state, timestep):
        p, m = atoms.get_momenta(), atoms.get_masses()[:, None]
        H, V = atoms.cell[:], atoms.get_volume()
        P = -reference_stress(atoms) + p.T @ (p / m) / V
        P_bias = sym(H.T @ prev["cell_bias"]) / V  # -σ_b, with σ_b = -(Hᵀ cell_bias) / V
        expected = P + P_bias - P_REF * onp.eye(3)
        assert onp.allclose(snapshot.extras["pressure"], expected, atol=1e-12)
        assert onp.allclose(state.xi, V)
        prev["cell_bias"] = onp.asarray(state.cell_bias)
        ncalls.append(1)

    make = INTEGRATORS[integrator]
    pysages.run(BOX_BIASES[method](), lambda **kw: make(atoms), nsteps, check)
    assert len(ncalls) == nsteps + 1


BIASED_CVS = {
    "Distance": ([Distance([0, 1])], 0.05, 3.0),
    "Distance+BoxVolume": ([Distance([0, 1]), BoxVolume()], onp.diag([0.05, 1e-3]), [3.0, 1200.0]),
}


@pytest.mark.parametrize("cvs", BIASED_CVS)
def test_bias_virial(cvs):
    # The barostat sees the biasing forces' virial, -(Xᵀ F_b) / V, and the box bias,
    # -(Hᵀ cell_bias) / V
    cvs, kspring, center = BIASED_CVS[cvs]
    atoms = lattice(T)
    method = HarmonicBias(cvs, kspring=kspring, center=center)
    result = pysages.run(method, lambda **kw: INTEGRATORS["NPTBerendsen"](atoms), 5)
    state = result.states[0]
    virial = atoms.get_positions().T @ onp.asarray(state.bias)
    if state.cell_bias is not None:
        virial += atoms.cell[:].T @ onp.asarray(state.cell_bias)
    assert onp.abs(virial).max() > 1e-4
    expected = reference_stress(atoms) - sym(virial) / atoms.get_volume()
    assert onp.allclose(atoms.get_stress(voigt=False), expected, atol=1e-14)


def test_barostat_feels_the_box_bias():
    # A bias toward a larger volume inflates the box, relative to the unbiased run (same seed)
    def berendsen(atoms):
        taut, taup, beta = 100 * units.fs, 200 * units.fs, 4.57e-5 / units.bar
        kwargs = dict(temperature_K=T, pressure_au=0.0, compressibility_au=beta)
        return NPTBerendsen(atoms, DT, taut=taut, taup=taup, **kwargs)

    V0 = 1.1 * lattice().get_volume()
    volumes = []
    biased = HarmonicBias([BoxVolume()], kspring=3e-5, center=V0)
    for method in (Unbiased([Component([0], 0)]), biased):
        atoms = lattice(T)
        pysages.run(method, lambda **kw: berendsen(atoms), 50)
        volumes.append(atoms.get_volume())
    assert volumes[1] > 1.02 * volumes[0]
