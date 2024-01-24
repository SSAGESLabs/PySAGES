---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="p49wJ0IjLAVD" -->

# Setup of the environment

<!-- #endregion -->

<!-- #region id="pF-5oR_GMkuI" -->

We download and install the environment of HOOMD-blue and OpenMM with their respective plugins.

<!-- #endregion -->

```bash id="nMThqa-DjVcb"

BASE_URL="https://drive.google.com/u/0/uc?id=1hsKkKtdxZTVfHKgqVF6qV2e-4SShmhr7&export=download"
wget -q --load-cookies /tmp/cookies.txt "$BASE_URL&confirm=$(wget -q --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $BASE_URL -O- | sed -rn 's/.*confirm=(\w+).*/\1\n/p')" -O pysages-env.zip
rm -rf /tmp/cookies.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="25H3kl03wzJe" outputId="55526734-bcea-4d1a-f1ae-de0b017126b7"
%env PYSAGES_ENV=/env/pysages
```

```bash id="V6MZXhOJMz7P"

mkdir -p $PYSAGES_ENV
unzip -qquo pysages-env.zip -d $PYSAGES_ENV
rm pysages-env.zip
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
First, we install the jaxlib version that matches the CUDA installation of this collab setup. See the JAX documentation [here](https://github.com/google/jax) for more details.

<!-- #endregion -->

```bash id="RUX1RAT3NF9s"

pip install -q --upgrade pip &> /dev/null
# Installs the wheel compatible with CUDA.
pip install -q --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html &> /dev/null
```

<!-- #region id="mx0IRythaTyG" -->

We test the jax installation and check the versions.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Z4E914qBHbZS" outputId="94391314-23b5-4726-f34a-fd927a0d4da1"
import jax
import jaxlib
print(jax.__version__)
print(jaxlib.__version__)
```

<!-- #region id="vtAmA51IAYxn" -->

Now we can finally install PySAGES. We clone the newest version from [here](https://github.com/SSAGESLabs/PySAGES) and build the remaining pure python dependencies and PySAGES itself.

<!-- #endregion -->

```bash id="rEsRX7GZNJ_R"

rm -rf PySAGES
git clone https://github.com/SSAGESLabs/PySAGES.git &> /dev/null
cd PySAGES
pip install -q . &> /dev/null
```

<!-- #region id="LjPjqVjSOzTL" -->

# Umbrella integration

In [this tutorial](https://github.com/SSAGESLabs/PySAGES/docs/notebooks/Harmonic_Bias_PySAGES_HOOMD.md), we demonstrated how PySAGES can be used to run a single simulation with a biasing potential.
However, if we want to look into the free-energy landscape a single simulation is not enough. Instead, we have to perform a series of simulations along a path in the space of the collective variables (CVs). From the histograms of the biasing, we can deduce the differences in free energy. For a more detailed explanation look at the literature, for example [J. Kaestner 2009](https://doi.org/10.1063/1.3175798).

The first step here is also to generate a simulation snapshot that can be used as an initial condition.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QOrufad1RaMF" outputId="02f68f5a-54cc-435c-e3df-78f4826dc374"
!pip install gsd &> /dev/null
import gsd
import gsd.hoomd
import numpy as np

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
    snapshot = gsd.hoomd.Frame()
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

system = System()
snap = get_snap(system)
snap = post_process_pos(snap)
snap.particles.validate()
with gsd.hoomd.open("start.gsd", "w") as f:
    f.append(snap)

```

<!-- #region id="AgFXHafmVUAi" -->

For this simulation, we are using the PySAGES method `UmbrellaIntegration` so we start with importing this.

In the next step, we write a function that generates the simulation context. We need to make sure that the context can depend on the replica of the simulation along the path. PySAGES sets variable `replica_num` in the keyword arguments of the function.
We also set some general parameters for all replicas.

In contrast to the single harmonic bias simulation, the simulation now contains an external potential `hoomd.external.periodic` which changes the expected density of particles. See hoomd-blue's [documentation](https://hoomd-blue.readthedocs.io/en/stable/module-md-external.html#hoomd.md.external.periodic) for details on the potential. For this example, the potential generates the free-energy landscape we are exploring.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tG6JhN7SNpSj" outputId="979d5793-f459-4202-ac4e-74f2aaabc1f3"
import hoomd
import hoomd.md
import hoomd.dlext

import pysages
from pysages.colvars import Component
from pysages.methods import UmbrellaIntegration
```

```python id="RsZhjfm2U5ps"
params = {"A": 0.5, "w": 0.2, "p": 2}

"""
Generates a simulation context, we pass this function to the attribute `run` of our sampling method.
"""
def generate_context(**kwargs):
    hoomd.context.initialize("")
    context = hoomd.context.SimulationContext()
    with context:
        print(f"Operating replica {kwargs.get('replica_num')}")
        system = hoomd.init.read_gsd("start.gsd")

        hoomd.md.integrate.nve(group=hoomd.group.all())
        hoomd.md.integrate.mode_standard(dt=0.01)

        nl = hoomd.md.nlist.cell()
        dpd = hoomd.md.pair.dpd(r_cut=1, nlist=nl, seed=42, kT=1.)
        dpd.pair_coeff.set("A", "A", A=5., gamma=1.0)
        dpd.pair_coeff.set("A", "B", A=5., gamma=1.0)
        dpd.pair_coeff.set("B", "B", A=5., gamma=1.0)

        periodic = hoomd.md.external.periodic()
        periodic.force_coeff.set('A', A=params["A"], i=0, w=params["w"], p=params["p"])
        periodic.force_coeff.set('B', A=0.0, i=0, w=0.02, p=1)
    return context

```

<!-- #region id="YRPnU0CJY31J" -->

With the ability to generate the simulation context, we start to set up the umbrella integration method - starting with the CV that describes the single A-particle along the varying axis of the external potential.

<!-- #endregion -->

```python id="_o7puY5Sao5h"
cvs = [Component([0], 0),]

```

<!-- #region id="jhs3vpglaux4" -->

Next, we define the path along the CV space. In this case, we start at position $-1.5$ and end the path at the position $1.5$. We are using linear interpolation with $25$ replicas.

<!-- #endregion -->

```python id="Uvkeedv4atn3"
centers = list(np.linspace(-1.5, 1.5, 25))
```

<!-- #region id="q37sUT-tbOMS" -->

The next parameters we need to define and run the method are the harmonic biasing spring constant,
(which we set to to $50$), the log frequency for the histogram ($50$), the number of steps we discard
as equilibration before logging ($10^3$), and the number of time steps per replica ($10^4$).

Since this runs multiple simulations, we expect the next cell to execute for a while.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wIrPB2N0bFIl" outputId="2f018685-a115-4c66-a21a-eef1d515bd02"
method = UmbrellaIntegration(cvs, 50.0, centers, 50, int(1e3))
raw_result = pysages.run(method, generate_context, int(1e4))
result = pysages.analyze(raw_result)
```

<!-- #region id="_xFSKCpKb6XF" -->

What is left after the run is evaluating the resulting histograms for each of the replicas. For a better visualization, we group the histogram into 4 separate plots. This also helps to demonstrate that the histograms overlap.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="OOpagwvlb3_d" outputId="62b507a1-d404-4924-ec1b-00b9d3f39085"
import matplotlib.pyplot as plt
bins =50
fig, ax = plt.subplots(2, 2)

counter = 0
hist_per = len(result["centers"])//4+1
for x in range(2):
    for y in range(2):
        for i in range(hist_per):
            if counter+i < len(result["centers"]):
                center = np.asarray(result["centers"][counter+i])
                histo, edges = result["histograms"][counter+i].get_histograms(bins=bins)
                edges = np.asarray(edges)[0]
                edges = (edges[1:] + edges[:-1]) / 2
                ax[x, y].plot(edges, histo, label=f"center {center}")
                ax[x, y].legend(loc="best", fontsize="xx-small")
                ax[x, y].set_yscale("log")
        counter += hist_per
while counter < len(result["centers"]):
    center = np.asarray(result["centers"][counter])
    histo, edges = result["histograms"][counter].get_histograms(bins=bins)
    edges = np.asarray(edges)[0]
    edges = (edges[1:] + edges[:-1]) / 2
    ax[1,1].plot(edges, histo, label=f"center {center}")
    counter += 1
```

<!-- #region id="5YZYZPUqdG7S" -->

And finally, as the last step, we can visualize the estimated free-energy path from the histograms and compare it with the analytical shape of the input external potential.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 297} id="_UKh6FyLcN9y" outputId="cba839f6-78e8-43c3-f540-5567c5c4b00e"
def external_field(r, A, p, w):
    return A * np.tanh(1 / (2 * np.pi * p * w) * np.cos(p * r))

fig, ax = plt.subplots()

ax.set_xlabel("CV")
ax.set_ylabel("Free energy $[\epsilon]$")
centers = np.asarray(result["centers"])
free_energy = np.asarray(result["free_energy"])
offset = np.min(free_energy)
ax.plot(centers, free_energy - offset, color="teal")

x = np.linspace(-2, 2, 50)
data = external_field(x, **params)
offset = np.min(data)
ax.plot(x, data - offset, label="test")

```

<!-- #region id="IXryBllMNiKM" -->

We can see, that the particle positions are indeed centered around the constraint values we set up earlier. Also, we see the shape of the histograms is very similar to the expected analytical prediction. We expect this since a liquid of soft particles is not that much different from an ideal gas.

<!-- #endregion -->
