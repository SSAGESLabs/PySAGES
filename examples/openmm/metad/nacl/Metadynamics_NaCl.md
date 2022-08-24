---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="T-Qkg9C9n7Cc" -->
# Setting up the environment

First, we set up our environment. We use an already compiled and packaged installation of OpenMM and the DLExt plugin.
We copy it from Google Drive and install PySAGES for it.
<!-- #endregion -->

```bash id="3eTbKklCnyd_"

BASE_URL="https://drive.google.com/u/0/uc?id=1hsKkKtdxZTVfHKgqVF6qV2e-4SShmhr7&export=download"
wget -q --load-cookies /tmp/cookies.txt "$BASE_URL&confirm=$(wget -q --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $BASE_URL -O- | sed -rn 's/.*confirm=(\w+).*/\1\n/p')" -O pysages-env.zip
rm -rf /tmp/cookies.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="KRPmkpd9n_NG" outputId="d1929eaa-42df-4a7d-afed-6094eab16e72"
%env PYSAGES_ENV=/env/pysages
```

```bash id="J7OY5K9VoBBh"

mkdir -p $PYSAGES_ENV .
unzip -qquo pysages-env.zip -d $PYSAGES_ENV
```

```python id="LlVSU_-FoD4w"
!update-alternatives --auto libcudnn &> /dev/null
```

```python id="EMAWp8VloIk4"
import os
import sys

ver = sys.version_info
sys.path.append(os.environ["PYSAGES_ENV"] + "/lib/python" + str(ver.major) + "." + str(ver.minor) + "/site-packages/")

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ["LD_LIBRARY_PATH"]
```

<!-- #region id="we_mTkFioS6R" -->
## PySAGES

The next step is to install PySAGES.
First, we install the jaxlib version that matches the CUDA installation of this Colab setup. See the JAX documentation [here](https://github.com/google/jax) for more details.
<!-- #endregion -->

```bash id="vK0RZtbroQWe"

pip install -q --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.0.5.
pip install -q --upgrade "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html &> /dev/null
```

<!-- #region id="wAtjM-IroYX8" -->
Now we can finally install PySAGES. We clone the newest version from [here](https://github.com/SSAGESLabs/PySAGES) and build the remaining pure python dependencies and PySAGES itself.
<!-- #endregion -->

```bash id="B-HB9CzioV5j"

rm -rf PySAGES
git clone https://github.com/SSAGESLabs/PySAGES.git &> /dev/null
cd PySAGES
pip install -q . &> /dev/null
```

<!-- #region id="KBFVcG1FoeMq" -->
# Metadynamics-biased simulations
<!-- #endregion -->

<!-- #region id="0W2ukJuuojAl" -->
Metadynamics gradually builds a biasing potential from a sum of gaussians that are deposited one at a time every certain number of (user defined) time steps.
There are two flavors of the algorithm, _Standard Metadynamics_ in which the heights of the gaussians is time independent, and _Well-tempered Metadynamics_ for which the heights of the deposited gaussians decreases depending on how frequently are visited the explored regions of collective variable space.

For this Colab, we are estimating the free energy along the distance between Na and Cl in water as example system.
<!-- #endregion -->

```bash id="fre3-LYso1hh"

# Download pdb file with the initial configuration of our system
PDB_URL="https://raw.githubusercontent.com/sivadasetty/PySAGES/nacl-example/examples/inputs/nacl/nacl-explicit.pdb"
wget -q $PDB_URL

# Download force field file 
ff_URL="https://raw.githubusercontent.com/openmm/openmmforcefields/main/amber/ffxml/tip3p_standard.xml"
wget -q $ff_URL
```

```python id="BBvC7Spoog82"
import numpy

from pysages.utils import try_import

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")


pi = numpy.pi

T = 298.15 * unit.kelvin
dt = 2.0 * unit.femtoseconds
nacl_pdb = "nacl-explicit.pdb"
nacl_ff = 'tip3p_standard.xml'


def generate_simulation(pdb_filename=nacl_pdb, T=T, dt=dt):
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField(nacl_ff)
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

    simulation.reporters.append(app.PDBReporter("output.pdb", 1000))
    simulation.reporters.append(
        app.StateDataReporter("log.dat", 1000, step=True, potentialEnergy=True, temperature=True)
    )

    return simulation
```

<!-- #region id="3UrzENm_oo6U" -->
Next, we load PySAGES and the relevant classes and methods for our problem
<!-- #endregion -->

```python id="fpMg-o8WomAA"
from pysages.grids import Grid
from pysages.colvars import Distance
from pysages.methods import Metadynamics

import pysages
```

<!-- #region id="LknkRvo1o4av" -->
The next step is to define the collective variable (CV). In this case, we choose the distance between Na and Cl.

For this example we will use the well-tempered version without grids. But these options can be configured.

We set the initial height, standard deviation and deposition frequency `stride` for the gaussians, as well as the $\Delta T$ Metadynamics parameter. And the number of time steps to run the simulation (here we use $5\times 10^5$ or 1 ns for the timestep chosen).

We also define a grid, which can be used as optional parameter to accelerate Metadynamics by approximating the biasing potential and its gradient by the closest value at the centers of the grid cells.

_Note:_ when setting $\Delta T$ we need to also provide a value for $k_B$ that matches the internal units used by the backend.
<!-- #endregion -->

```python id="B1Z8FWz0o7u_"
# IDs of Na and Cl using PDB entry (subtract 1 as we use zero-based indexing)
cvs = [Distance([1524, 1525])] 

well_tempered = True  # False for standard metadynamics
use_grids = False

height = 1.2  # kJ/mol
sigma = [0.05]  # nm
deltaT = 5000 if well_tempered else None
stride = 500  # frequency for depositing gaussians
timesteps = 500000
ngauss = timesteps // stride + 1  # total number of gaussians

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * unit.kelvin
kB = kB.value_in_unit(unit.kilojoules_per_mole)  # unitless kB value

# 1D Grid for storing bias potential and its gradient
grid = pysages.Grid(lower=(0), upper=(4), shape=(500), periodic=False)
grid = grid if use_grids else None

method = Metadynamics(cvs, height, sigma, stride, ngauss, deltaT=deltaT, kB=kB, grid=grid)
```

<!-- #region id="Fz8BfU34pA_N" -->
We now simulate the number of time steps set above.
Make sure to run with GPU support, otherwise, it can take a very long time.
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
metapotential = fe_result['metapotential']
```

<!-- #region id="VfTQ5yeyxt8e" -->
Next we use the biasing potential to estimate the free energy surface. For well-tempered metadynamics this is equal to the sum of accumulated gaussians scaled by the factor $-(T + \Delta T)\, / \,\Delta T$.
<!-- #endregion -->

```python id="6W7Xf0ilqAcm"
# generate CV values on a grid to evaluate bias potential
plot_grid = pysages.Grid(lower=(0), upper=(4), shape=(500), periodic=False)
xi = (compute_mesh(plot_grid) + 1) / 2 * plot_grid.size + plot_grid.lower
xi = xi.flatten()

alpha = 1 if method.deltaT is None else (T.value_in_unit(unit.kelvin) + method.deltaT) / method.deltaT
kT = kB * T.value_in_unit(unit.kelvin)

A = metapotential(xi) * -alpha / kT
A = A - A.min()
A = A.reshape(plot_grid.shape)
```

<!-- #region id="Kf_CMdih90Cd" -->
And plot it.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="3s9LL9apBMVb" outputId="6384612f-d75a-4541-b988-c4660ab2e570"
fig, ax = plt.subplots(dpi=120)

ax.plot(xi, A, lw=5, alpha=0.75, color="darkred")
ax.set_xlabel(r"$r~[nm]$")
ax.set_ylabel(r"$A~[k_{B}T]$")
ax.set_xlim(0.1,1.25)
ax.set_ylim(0.1,10)

#fig.savefig("nacl-fe.png", dpi=fig.dpi)
plt.show()
```

<!-- #region id="a-LGmeZ_3_m-" -->
Lastly, we plot the height of the gaussians as a function of time and observe that their height decreases at an exponential rate as expected for well-tempered metadynamics.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 457} id="SI_fhUW9CGlP" outputId="82a23a39-2cd9-46ab-df80-fe6ac9a7cbf6"
_dt = dt #method.context[0].sampler.snapshot.dt
ts = _dt * 1e-3 * numpy.arange(0, fe_result['heights'].size) * run_result.method.stride

fig, ax = plt.subplots(dpi=120)
ax.plot(ts, fe_result['heights'], "o", mfc="none", ms=4)
ax.set_xlabel("time [ns]")
ax.set_ylabel("height [kJ/mol]")
plt.show()
```

```python id="R6rEuwWAZ8Qp"

```
