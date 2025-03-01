---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="_UgEohXC8n0g" -->
# Setting up the environment

First, we set up our environment. We will be using a pre-compiled and packaged installation of OpenMM and the openmm-dlext plugin.
It will be downloaded from Google Drive and made accessible to the Python process running in this Colab instance.
<!-- #endregion -->

```bash id="3eTbKklCnyd_"

BASE_URL="https://drive.usercontent.google.com/download?id=1hsKkKtdxZTVfHKgqVF6qV2e-4SShmhr7"
COOKIES="/tmp/cookies.txt"
CONFIRMATION="$(wget -q --save-cookies $COOKIES --keep-session-cookies --no-check-certificate $BASE_URL -O- | sed -rn 's/.*confirm=(\w+).*/\1\n/p')"

wget -q --load-cookies $COOKIES "$BASE_URL&confirm=$CONFIRMATION" -O pysages-env.zip
rm -rf $COOKIES
```

```python colab={"base_uri": "https://localhost:8080/"} id="25H3kl03wzJe" outputId="528d12be-8cc4-42d9-d460-692d46a0e662"
%env PYSAGES_ENV=/env/pysages
```

```bash id="CPkgxfj6w4te"

mkdir -p $PYSAGES_ENV .
unzip -qquo pysages-env.zip -d $PYSAGES_ENV
```

```python id="JMO5fiRTxAWB"
import os
import sys

ver = sys.version_info
sys.path.append(os.environ["PYSAGES_ENV"] + "/lib/python" + str(ver.major) + "." + str(ver.minor) + "/site-packages/")

os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ["LD_LIBRARY_PATH"]
```

<!-- #region id="lf2KeHt5_eFv" -->
## PySAGES

Next, we install PySAGES. The latest version is retrieved from GitHub and installed (along with its dependencies) using `pip`.
<!-- #endregion -->

```python id="B-HB9CzioV5j"
!pip install -qq git+https://github.com/SSAGESLabs/PySAGES.git > /dev/null
```

<!-- #region id="KBFVcG1FoeMq" -->
# Metadynamics-biased simulations
<!-- #endregion -->

<!-- #region id="0W2ukJuuojAl" -->

Metadynamics gradually builds a biasing potential from a sum of gaussians that are deposited one at a time every certain number of (user defined) time steps.
There are two flavors of the algorithm, _Standard Metadynamics_ in which the heights of the gaussians is time independent, and _Well-tempered Metadynamics_ for which the heights of the deposited gaussians decreases depending on how frequently are visited the explored regions of collective variable space.

For this Colab, we are using alanine peptide in vacuum as example system.

<!-- #endregion -->

```bash id="fre3-LYso1hh"

# Download pdb file with the initial configuration of our system
wget -q https://raw.githubusercontent.com/SSAGESLabs/PySAGES/main/examples/inputs/alanine-dipeptide/adp-vacuum.pdb
```

```python id="BBvC7Spoog82"
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit


pi = np.pi
T = 298.15 * unit.kelvin
dt = 2.0 * unit.femtoseconds
adp_pdb = "adp-vacuum.pdb"


def generate_simulation(pdb_filename=adp_pdb, T=T, dt=dt, **kwargs):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml")
    cutoff_distance = 1.0 * unit.nanometer
    topology = pdb.topology

    system = ff.createSystem(
        topology, constraints=app.HBonds, nonbondedMethod=app.PME, nonbondedCutoff=cutoff_distance
    )

    # Set dispersion correction use.
    forces = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        forces[force.__class__.__name__] = force

    forces["NonbondedForce"].setUseDispersionCorrection(True)
    forces["NonbondedForce"].setEwaldErrorTolerance(1.0e-5)

    positions = pdb.getPositions(asNumpy=True)

    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(42)

    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    return simulation
```

<!-- #region id="3UrzENm_oo6U" -->

Next, we import PySAGES and the necessary classes and methods for our simulation

<!-- #endregion -->

```python id="fpMg-o8WomAA"
import pysages

from pysages.grids import Grid
from pysages.colvars import DihedralAngle
from pysages.methods import Metadynamics
```

<!-- #region id="LknkRvo1o4av" -->
The next step is to define the collective variable (CV). In this case, we choose the so called $\phi$ and $\psi$ dihedral angles of alanine dipeptide.

For this example we will use the well-tempered version without grids. But these options can be configured.

We set the initial height, standard deviation and deposition frequency `stride` for the gaussians, as well as the $\Delta T$ Metadynamics parameter. And the number of time steps to run the simulation (here we use $9\times10^6$ or 18 ns for the timestep chosen).

We also define a grid, which can be used as optional parameter to accelerate Metadynamics by approximating the biasing potential and its gradient by the closest value at the centers of the grid cells.

_Note:_ when setting $\Delta T$ we need to also provide a value for $k_B$ that matches the internal units used by the backend.
<!-- #endregion -->

```python id="B1Z8FWz0o7u_"
cvs = [DihedralAngle([4, 6, 8, 14]), DihedralAngle([6, 8, 14, 16])]

well_tempered = True  # False for standard metadynamics
use_grids = False

height = 1.2  # kJ/mol
sigma = [0.35, 0.35]  # radians
deltaT = 5000 if well_tempered else None
stride = 200  # frequency for depositing gaussians
timesteps = int(1e6)
ngauss = timesteps // stride  # total number of gaussians

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * unit.kelvin
kB = kB.value_in_unit(unit.kilojoules_per_mole)  # unitless kB value

# Grid for partitioning the bias potential and its gradient
grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(64, 64), periodic=True)
grid = grid if use_grids else None

method = Metadynamics(cvs, height, sigma, stride, ngauss, deltaT=deltaT, kB=kB, grid=grid)
```

<!-- #region id="Fz8BfU34pA_N" -->
We now simulate the number of time steps set above.
For optimal performance, ensure that the simulation is executed with GPU acceleration. Otherwise, it may take a considerably longer time to complete.
On the GPU this should run in around half an hour.
<!-- #endregion -->

```python id="K951m4BbpUar"
run_result = pysages.run(method, generate_simulation, timesteps)
```

<!-- #region id="PXBKUfK0p9T2" -->
## Analysis

Let's plot the negative of the sum of gaussians accumulated. This will get close to the free energy surface for long enough simulations (larger than what is practical to run on Colab, but we should get close enough for illustration purposes here).
<!-- #endregion -->

```python id="X69d1R7OpW4P"
import matplotlib.pyplot as plt
from pysages.approxfun import compute_mesh
```

<!-- #region id="6mrlIOfoszBJ" -->
We are now going to gather the information for the heights, standard deviations and centers of the accumulated gaussians and build a function to evaluate their sum at any point of the collective variable space.
<!-- #endregion -->

```python id="zJqvpbw8szxR"
fe_result = pysages.analyze(run_result)
metapotential = fe_result["metapotential"]
```

<!-- #region id="VfTQ5yeyxt8e" -->

Next we use the biasing potential to estimate the free energy surface. For well-tempered metadynamics this is equal to the sum of accumulated gaussians scaled by the factor $-(T + \Delta T)\, / \,\Delta T$.

<!-- #endregion -->

```python id="6W7Xf0ilqAcm"
plot_grid = pysages.Grid(lower=(-pi, -pi), upper=(pi, pi), shape=(64, 64), periodic=True)
xi = (compute_mesh(plot_grid) + 1) / 2 * plot_grid.size + plot_grid.lower

alpha = 1 if method.deltaT is None else (T.value_in_unit(unit.kelvin) + method.deltaT) / method.deltaT
kT = kB * T.value_in_unit(unit.kelvin)

A = metapotential(xi) * -alpha / kT
A = A - A.min()
A = A.reshape(plot_grid.shape)
```

<!-- #region id="Kf_CMdih90Cd" -->
And plot it.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 461} id="3s9LL9apBMVb" outputId="55abf4e5-fef0-4faa-bf01-9719cbe8aa2b"
fig, ax = plt.subplots(dpi=120)

im = ax.imshow(
    A, interpolation="bicubic", origin="lower", extent=[-pi, pi, -pi, pi]
)
ax.contour(
    A, levels=12, linewidths=0.75, colors="k", extent=[-pi, pi, -pi, pi]
)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\psi$")

cbar = plt.colorbar(im)
cbar.ax.set_ylabel(r"$A~[k_{B}T]$", rotation=270, labelpad=20)

plt.show()
```

<!-- #region id="a-LGmeZ_3_m-" -->
Lastly, we plot the height of the gaussians as a function of time and observe that their height decreases at an exponential rate as expected for well-tempered metadynamics.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 457} id="SI_fhUW9CGlP" outputId="5d32f99d-4911-44bb-9d89-69c3e6212cb7"
ts = dt * 1e-3 * np.arange(0, fe_result["heights"].size) * run_result.method.stride

fig, ax = plt.subplots(dpi=120)

ax.set_xlabel("time [ns]")
ax.set_ylabel("height [kJ/mol]")
ax.plot(ts, fe_result["heights"], "o", mfc="none", ms=4)

fig.show()
```
