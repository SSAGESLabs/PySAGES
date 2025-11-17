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

<!-- #region id="fRZDARPsDQHF" -->
# Harmonic Bias simulation
<!-- #endregion -->

<!-- #region id="Uh2y2RXDDZub" -->
A harmonic bias simulation constraints a collective variable with a harmonic potential. This is useful for a variety of advanced sampling methods, in particular, umbrella sampling.

For this Colab, we are generating a small system of soft DPD particles first. This system of soft particles allows fast reliable execution.
For this, we use the [GSD](https://gsd.readthedocs.io/en/stable/) file format and its python frontend to generate the initial conditions.
Since all particles are soft, it is OK to start with random positions inside the simulation box. We also assign random velocities drawn from the Maxwell-Boltzmann distribution. The final configuration is written to disk and can be opened by HOOMD-blue for simulations.
<!-- #endregion -->

```python id="aIP9vx8yDdr1"
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
    snapshot.particles.types = ["A"]
    snapshot.particles.position = np.zeros((N, 3))
    snapshot.particles.velocity = np.random.standard_normal((N, 3))
    snapshot.particles.image = np.zeros((N, 3), dtype=int)
    snapshot.particles.typeid = np.zeros(N, dtype=int)

    rng = np.random.default_rng()
    for particle in range(N):
        snapshot.particles.position[particle, 0] = (rng.random() * L - L / 2)
        snapshot.particles.position[particle, 1] = (rng.random() * L - L / 2)
        snapshot.particles.position[particle, 2] = (rng.random() * L - L / 2)

    return snapshot


system = System()
snap = get_snap(system)
snap = post_process_pos(snap)
snap.particles.validate()

with gsd.hoomd.open("harmonic_start.gsd", "w") as f:
   f.append(snap)
```

<!-- #region id="n0Rd-hMnCD-B" -->
Next, we start running the system, we start with importing the required libraries.
Noteworthy are here the hoomd and the pysages package.

We are going to use a collective variable that constrains a particle position.
In PySAGES the `Component` class from the `colvars` package can achieve this for us.

The `HarmonicBias` class is responsible for introducing the bias into the simulation run,
while `HistogramLogger` collects the state of the collective variable during the run.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HkHOzXMzExps" outputId="27c1f5c0-43d4-4911-f1f8-069709242593"
import hoomd

import pysages
from pysages.colvars import Component
from pysages.methods import HarmonicBias, HistogramLogger
```

<!-- #region id="YibErIQhC0Lv" -->
The next step is to write a function that generates the simulation context.
Inside this function is the HOOMD-blue specific code, that you would normally write to execute a HOOMD-blue simulation. Here it is packaged into a function, such that PySAGES can deploy the simulation context when needed.
In this case, we use the GSD file read in the initial, and define the DPD forcefield with parameters.
DPD is a special case in HOOMD-blue. The thermostat is part of the pair-potential and not part of the integrator. Hence, we specify NVE integration and all thermostat parameter for NVT in the potential. The function returns the simulation context for PySAGES to work with.

The second function is a helper function to generate the theoretically expected distribution of a harmonically biased simulation of an ideal gas in NVT. And helps to verify the results of the simulation.
<!-- #endregion -->

```python id="67488aXwQXba"
def generate_simulation(
    kT=1, dt=0.01, A=5, gamma=1, r_cut=1,
    device=hoomd.device.auto_select(), seed=42,
    **kwargs
):
    """
    Generates a simulation context to which will attatch our sampling method.
    """
    simulation = hoomd.Simulation(device=device, seed=seed)
    simulation.create_state_from_gsd("harmonic_start.gsd")
    simulation.run(0)

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPD(nlist=nlist, kT=kT, default_r_cut=r_cut)
    dpd.params[("A", "A")] = dict(A=A, gamma=gamma)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())

    integrator = hoomd.md.Integrator(dt=dt)
    integrator.forces.append(dpd)
    integrator.methods.append(nve)
    simulation.operations.integrator = integrator

    return simulation


def get_target_dist(center, k, lim, bins):
    x = np.linspace(lim[0], lim[1], bins)
    p = np.exp(-0.5 * k * (x - center)**2)
    # norm numerically
    p *= (lim[1] - lim[0]) / np.sum(p)
    return p
```

<!-- #region id="BgQ88M0sIfbp" -->
The next step is to define the collective variables (CVs) we are interested in.
In this case, we are using the `Component` CV to describe the position in space. We choose particle `[0]` for this and log in 3 different CVS the Z- `2`, Y- `1`, and X- `0` position of the particle.
The center describes where we are restraining the CVs to, which is also specified for each of the CVs described earlier.

Finally, we define the spring constant for the harmonic biasing potential and the `HarmonicBias` method itself.
<!-- #endregion -->

```python id="r911REinQdLF"
cvs = [Component([0], 2), Component([0], 1), Component([0], 0)]
cv_centers = [0.0, 1.0, -0.3]
k = 15
method = HarmonicBias(cvs, k, cv_centers)
```

<!-- #region id="bGIDE56RLCcP" -->
Next, we define the `HistogramLogger` callback. The callback interacts with the simulation every timestep after the biasing. In this case, we use it to log the state of the collective variables every `100` time-steps.

And we can finally run the simulations. This happens through the PySAGES method run and is transparent to the user which backend is running.
Here, the run is just a simple simulation for the number of steps specified with the biasing potential. Other advanced sampling methods can have more advanced run schemes.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aOXCppWkQnJI" outputId="a34ae4f3-92a9-47ce-cac6-7a02f1aa4a72"
callback = HistogramLogger(100)
pysages.run(method, generate_simulation, int(1e4), callback, {"A": 7.0})
```

<!-- #region id="_vigR7XaMUD3" -->
After the simulation run, we collect the results for comparison with the analytic prediction for an ideal gas.
First, we generate the analytic predictions for each of the CVs in a list `target_hist`.

After that, we are using the collected results from the callback to build the histograms from the simulations, and store the results in `hist_list`.
<!-- #endregion -->

```python id="jBiATDSaSqUw"
Lmax = 5.0
bins = 25
target_hist = []

for i in range(len(center_cv)):
    target_hist.append(
        get_target_dist(center_cv[i], k, (-Lmax / 2, Lmax / 2), bins)
    )

lims = [(-Lmax / 2, Lmax / 2) for i in range(3)]
hist, edges = callback.get_histograms(bins=bins, range=lims)
hist_list = [
    np.sum(hist, axis=(1, 2)) / (Lmax ** 2),
    np.sum(hist, axis=(0, 2)) / (Lmax ** 2),
    np.sum(hist, axis=(0, 1)) / (Lmax ** 2),
]
lim = (-Lmax / 2, Lmax / 2)
```

<!-- #region id="2xwriftjNKgz" -->
Finally, we want to evaluate how the simulations turned out.
We use matplotlib to visualize the expected (dashed) and actual results of the simulations (solid).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 301} id="ZCkylgdvS3To" outputId="440269b2-ef60-4bce-b9fc-f34c823c8299"
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.set_xlabel(r"CV $\xi_i$")
ax.set_ylabel(r"$p(\xi_i)$")

x = np.linspace(lim[0], lim[1], hist_list[0].shape[0])

for i in range(len(hist_list)):
    (line,) = ax.plot(x, hist_list[i], label="i= {0}".format(i))
    ax.plot(x, target_hist[i], "--", color=line.get_color())

ax.legend(loc="best")

fig.show()
```

<!-- #region id="IXryBllMNiKM" -->
We can see, that the particle positions are indeed centered around the constraints we set up earlier. Also, we see the shape of the histograms is very similar to the expected analytical prediction. We expect this since a liquid of soft particles is not that much different from an ideal gas.
<!-- #endregion -->
