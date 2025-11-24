---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.0.dev0
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="T-Qkg9C9n7Cc" -->
# Setting up the environment

First, we set up our environment. We will be using a pre-compiled and packaged installation of HOOMD-blue and the hoomd-dlext plugin.
It will be downloaded from Google Drive and made accessible to the Python process running in this Colab instance.
<!-- #endregion -->

```bash id="3eTbKklCnyd_"

BASE_URL="https://drive.usercontent.google.com/download?id=1hsKkKtdxZTVfHKgqVF6qV2e-4SShmhr7"
COOKIES="/tmp/cookies.txt"
CONFIRMATION="$(wget -q --save-cookies $COOKIES --keep-session-cookies --no-check-certificate $BASE_URL -O- | sed -rn 's/.*confirm=(\w+).*/\1\n/p')"

wget -q --load-cookies $COOKIES "$BASE_URL&confirm=$CONFIRMATION" -O pysages-env.zip
rm -rf $COOKIES
```

```python colab={"base_uri": "https://localhost:8080/"} id="KRPmkpd9n_NG" outputId="b757f2aa-38cc-4726-c4ab-5197810b9d77"
%env PYSAGES_ENV=/env/pysages
```

```bash id="J7OY5K9VoBBh"

mkdir -p $PYSAGES_ENV .
unzip -qquo pysages-env.zip -d $PYSAGES_ENV
```

```python id="EMAWp8VloIk4"
import os
import sys

ver = sys.version_info
sys.path.append(os.environ["PYSAGES_ENV"] + "/lib/python" + str(ver.major) + "." + str(ver.minor) + "/site-packages/")

os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ["LD_LIBRARY_PATH"]
```

<!-- #region id="Wy-75Pt7Bqs1" -->
We'll also need some additional python dependencies
<!-- #endregion -->

```python id="LpBucu3V81xm"
!pip install -qq "numpy<2" gsd > /dev/null
```

<!-- #region id="we_mTkFioS6R" -->
## PySAGES

Next, we install PySAGES. The latest version is retrieved from GitHub and installed (along with its dependencies) using `pip`.
<!-- #endregion -->

```python id="B-HB9CzioV5j"
!pip install -qq git+https://github.com/SSAGESLabs/PySAGES.git > /dev/null
```

<!-- #region id="LjPjqVjSOzTL" -->
# Umbrella integration

In [this tutorial](https://github.com/SSAGESLabs/PySAGES/blob/main/examples/hoomd-blue/harmonic_bias/Harmonic_Bias.md), we demonstrated how PySAGES can be used to run a single simulation with a biasing potential.
However, if we want to look into the free-energy landscape a single simulation is not enough. Instead, we have to perform a series of simulations along a path in the space of the collective variables (CVs). From the histograms of the biasing, we can deduce the differences in free energy. For a more detailed explanation look at the literature, for example [J. Kaestner 2009](https://doi.org/10.1063/1.3175798).

The first step here is also to generate a simulation snapshot that can be used as an initial condition.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QOrufad1RaMF" outputId="02f68f5a-54cc-435c-e3df-78f4826dc374"
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

In contrast to the single harmonic bias simulation, the simulation now contains an external potential `hoomd.md.external.field.Periodic` which changes the expected density of particles. See hoomd-blue's [documentation](https://hoomd-blue.readthedocs.io/en/stable/module-md-external-field.html) for details on the potential. For this example, the potential generates the free-energy landscape we are exploring.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tG6JhN7SNpSj" outputId="979d5793-f459-4202-ac4e-74f2aaabc1f3"
import hoomd

import pysages
from pysages.colvars import Component
from pysages.methods import UmbrellaIntegration
```

```python id="RsZhjfm2U5ps"
dpd_params = dict(
    AA = dict(A = 5, gamma = 1),
    AB = dict(A = 5, gamma = 1),
    BB = dict(A = 5, gamma = 1),
)
periodic_params = dict(
    A = dict(A = 0.5, i = 0, w = 0.2, p = 2),
    B = dict(A = 0.0, i = 0, w = 0.02, p = 1),
)

def generate_simulation(
    kT=1, dt=0.01, r_cut=1, dpd_params=dpd_params, periodic_params=periodic_params,
    device=hoomd.device.auto_select(), seed=42,
    **kwargs
):
    """
    Generates a simulation context to which will attatch our sampling method.
    """
    print(f"Operating replica {kwargs.get('replica_num')}")

    simulation = hoomd.Simulation(device=device, seed=seed)
    simulation.create_state_from_gsd("start.gsd")
    simulation.run(0)

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPD(nlist=nlist, kT=kT, default_r_cut=r_cut)
    dpd.params[("A", "A")] = dpd_params["AA"]
    dpd.params[("A", "B")] = dpd_params["AB"]
    dpd.params[("B", "B")] = dpd_params["BB"]

    periodic = hoomd.md.external.field.Periodic()
    periodic.params["A"] = periodic_params["A"]
    periodic.params["B"] = periodic_params["B"]

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())

    integrator = hoomd.md.Integrator(dt=dt)
    integrator.forces.append(dpd)
    integrator.methods.append(nve)
    simulation.operations.integrator = integrator

    return simulation
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
(which we set to to $100$), the log frequency for the histogram ($50$), the number of steps we discard
as equilibration before logging ($10^3$), and the number of time steps per replica ($10^4$).

Since this runs multiple simulations, we expect the next cell to execute for a while.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wIrPB2N0bFIl" outputId="2f018685-a115-4c66-a21a-eef1d515bd02"
method = UmbrellaIntegration(cvs, 100.0, centers, 50, int(1e3))
raw_result = pysages.run(method, generate_simulation, int(1e4))
result = pysages.analyze(raw_result)
```

<!-- #region id="_xFSKCpKb6XF" -->
What is left after the run is evaluating the resulting histograms for each of the replicas. For a better visualization, we group the histogram into 4 separate plots. This also helps to demonstrate that the histograms overlap.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="OOpagwvlb3_d" outputId="62b507a1-d404-4924-ec1b-00b9d3f39085"
import matplotlib.pyplot as plt

bins = 50

fig, ax = plt.subplots(2, 2)

counter = 0
hist_per = len(result["centers"]) // 4 + 1

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
def external_field(r, A, p, w, **kwargs):
    return A * np.tanh(1 / (2 * np.pi * p * w) * np.cos(p * r))

x = np.linspace(-2, 2, 100)
data = external_field(x, **periodic_params["A"])

centers = np.asarray(result["centers"])
free_energy = np.asarray(result["free_energy"])

fig, ax = plt.subplots()

ax.set_xlabel("CV")
ax.set_ylabel("Free energy $[\epsilon]$")
ax.plot(x, data - np.min(data), label="test")
ax.plot(centers, free_energy - np.min(free_energy), color="teal")

fig.show()
```

<!-- #region id="IXryBllMNiKM" -->
We can see, that the particle positions are indeed centered around the constraint values we set up earlier. Also, we see the shape of the histograms is very similar to the expected analytical prediction. We expect this since a liquid of soft particles is not that much different from an ideal gas.
<!-- #endregion -->
