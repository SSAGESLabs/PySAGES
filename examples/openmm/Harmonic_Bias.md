---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="_UgEohXC8n0g" -->

# Setting up the environment

First, we are setting up our environment. We use an already compiled and packaged installation of OpenMM and the DLExt plugin. We copy it from Google Drive and install PySAGES for it. We also have a Google Colab that performs this installation for reference, but that requires permissions that we do not want on our Google Drive.

<!-- #endregion -->

```bash id="nMThqa-DjVcb"

BASE_URL="https://drive.google.com/u/0/uc?id=1hsKkKtdxZTVfHKgqVF6qV2e-4SShmhr7&export=download"
wget -q --load-cookies /tmp/cookies.txt "$BASE_URL&confirm=$(wget -q --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $BASE_URL -O- | sed -rn 's/.*confirm=(\w+).*/\1\n/p')" -O pysages-env.zip
rm -rf /tmp/cookies.txt
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
```

<!-- #region id="lf2KeHt5_eFv" -->

## PySAGES

The next step is to install PySAGES.
First, we install the jaxlib version that matches the CUDA installation of this Colab setup. See the JAX documentation [here](https://github.com/google/jax) for more details.

<!-- #endregion -->

<!-- #region id="mx0IRythaTyG" -->

We test the jax installation and check the versions.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Z4E914qBHbZS" outputId="56c47936-19c1-4de8-fbc7-1cace7282498"
import jax
import jaxlib
print(jax.__version__)
print(jaxlib.__version__)
```

<!-- #region id="vtAmA51IAYxn" -->

Now we can finally install PySAGES. We clone the newest version from [here](https://github.com/SSAGESLabs/PySAGES) and build the remaining pure python dependencies and PySAGES itself.

<!-- #endregion -->

```bash id="xYRGOcFJjEE6"

rm -rf PySAGES
git clone https://github.com/SSAGESLabs/PySAGES.git &> /dev/null
cd PySAGES
pip install -q . &> /dev/null
```

<!-- #region id="h5xD1zfj-J2z" -->

# Harmonic Bias simulations

<!-- #endregion -->

```bash id="OIyRfOU9_cEJ"

mkdir /content/harmonic-bias
cd /content/harmonic-bias
```

<!-- #region id="Uh2y2RXDDZub" -->

A harmonic bias simulation constraints a collective variable with a harmonic potential. This is useful for a variety of advanced sampling methods, in particular, umbrella sampling.

For this Colab, we are using alanine dipeptide as the example molecule, a system widely-used for benchmarking enhanced sampling methods. So first, we fetch the molecule from the examples of PySAGES.

<!-- #endregion -->

```bash id="5fxJMNyE-RdA"

cp /content/PySAGES/examples/inputs/alanine-dipeptide/adp-explicit.pdb ./
```

<!-- #region id="SqaG8YdK__FU" -->

Next we load the PySAGES/OpenMM environment.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="P6kPLtGI_-66" outputId="98e496cb-b78d-47bf-8b96-f2af942b10fc"
from pysages.colvars import DihedralAngle
from pysages.methods import HarmonicBias, HistogramLogger
import numpy as np
from pysages.utils import try_import

import pysages

openmm = try_import("openmm", "simtk.openmm")
unit = try_import("openmm.unit", "simtk.unit")
app = try_import("openmm.app", "simtk.openmm.app")
```

<!-- #region id="3TV4h_WEAdSm" -->

Next, we write a function that can generate an execution context for OpenMM. This is everything you would normally write in an OpenMM script, just wrapped as a function that returns the context of the simulation.

<!-- #endregion -->

```python id="GAGw0s_cAcgP"
"""
Generates a simulation context, we pass this function to the attribute `run` of our sampling method.
"""
def generate_simulation(**kwargs):
    pdb_filename = "adp-explicit.pdb"
    T = 298.15 * unit.kelvin
    dt = 2.0 * unit.femtoseconds
    pdb = app.PDBFile(pdb_filename)

    ff = app.ForceField("amber99sb.xml", "tip3p.xml")
    cutoff_distance = 1.0 * unit.nanometer
    topology = pdb.topology
    system = ff.createSystem(
        topology, constraints = app.HBonds, nonbondedMethod = app.NoCutoff,
        nonbondedCutoff = cutoff_distance
    )

    positions = pdb.getPositions(asNumpy = True)

    integrator = openmm.LangevinIntegrator(T, 1 / unit.picosecond, dt)

    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    return simulation
```

<!-- #region id="YtUoUMEdKtH8" -->

The next step is to define the collective variable (CV). In this case, we choose the two dihedral angles on the molecule as defined by the atom positions. We also choose an equilibrium value to constrain the dihedrals and the corresponding spring constant.
The `HarmonicBias` class is responsible for introducing the bias into the simulation run.

<!-- #endregion -->

```python id="zEH5jrRoKszT"
cvs = (DihedralAngle((4, 6, 8, 14)), DihedralAngle((6, 8, 14, 16)))
center =[ -0.33*np.pi, -0.4*np.pi]
k = 100
method = HarmonicBias(cvs, k, center)
```

<!-- #region id="sqKuZo92K9n9" -->

We now define a Histogram callback to log the measured values of the CVs and run the simulation for $10^4$ time steps. The `HistogramLogger` collects the state of the collective variable during the run.
Make sure to run with GPU support. Using the CPU platform with OpenMM is possible and supported, but can take a very long time.

<!-- #endregion -->

```python id="-XKSe3os_-Rg"
callback = HistogramLogger(50)
pysages.run(method, generate_simulation, int(1e4), callback)
```

<!-- #region id="z8V0iX70RF1m" -->

Next, we want to plot the histogram as recorded from the simulations.

<!-- #endregion -->

```python id="Mvq9CWdg_qxl"
bins = 25
lim = (-np.pi/2, -np.pi/4)
lims = [lim for i in range(2)]
hist, edges = callback.get_histograms(bins=bins, range=lims)
hist_list = [np.sum(hist, axis=(0)), np.sum(hist, axis=(1))]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 301} id="mxZVBr2FR5FJ" outputId="2d0d189b-a1b8-400d-92cd-0fbbeaa783e8"
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.set_xlabel(r"CV $\xi_i$")
ax.set_ylabel(r"$p(\xi_i)$")

x = np.linspace(lim[0], lim[1], hist_list[0].shape[0])

for i in range(len(hist_list)):
    (line,) = ax.plot(x, hist_list[i], label="i= {0}".format(i))

ax.legend(loc="best")
```

<!-- #region id="m9JjGXq_ha-6" -->

We see how the dihedral angles are distributed. The histograms are not perfect in this example because we ran the simulation only for a few time steps and hence sampling is quite limited.

<!-- #endregion -->
