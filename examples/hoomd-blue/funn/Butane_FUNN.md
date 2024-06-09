---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="T-Qkg9C9n7Cc" -->

# Setting up the environment

First, we are setting up our environment. We use an already compiled and packaged installation of HOOMD-blue and the DLExt plugin.
We copy it from Google Drive and install PySAGES for it.

<!-- #endregion -->

```bash id="3eTbKklCnyd_"

BASE_URL="https://drive.google.com/u/0/uc?id=1hsKkKtdxZTVfHKgqVF6qV2e-4SShmhr7&export=download"
wget -q --load-cookies /tmp/cookies.txt "$BASE_URL&confirm=$(wget -q --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $BASE_URL -O- | sed -rn 's/.*confirm=(\w+).*/\1\n/p')" -O pysages-env.zip
rm -rf /tmp/cookies.txt
```

```python colab={"base_uri": "https://localhost:8080/"} id="KRPmkpd9n_NG" outputId="2c72bbc3-0731-4d62-98e1-d48f8254adcb"
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
# Installs the wheel compatible with CUDA.
pip install -q --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html &> /dev/null
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

```bash id="ppTzMmyyobHB"

mkdir /content/funn
cd /content/funn
```

<!-- #region id="KBFVcG1FoeMq" -->

# FUNN-biased simulations

<!-- #endregion -->

<!-- #region id="0W2ukJuuojAl" -->

FUNN gradually learns the free energy gradient from a discrete estimate based on the same algorithm as the ABF method, but employs a neural network to provide a continuous approximation to it.

For this Colab, we are using butane as the example molecule.

<!-- #endregion -->

```python id="BBvC7Spoog82"
import hoomd
import hoomd.md

import numpy


pi = numpy.pi
kT = 0.596161
dt = 0.02045
mode = "--mode=gpu"


def generate_context(kT = kT, dt = dt, mode = mode):
    """
    Generates a simulation context, we pass this function to the attribute
    `run` of our sampling method.
    """
    hoomd.context.initialize(mode)

    ### System Definition
    snapshot = hoomd.data.make_snapshot(
        N = 14,
        box = hoomd.data.boxdim(Lx = 41, Ly = 41, Lz = 41),
        particle_types = ['C', 'H'],
        bond_types = ["CC", "CH"],
        angle_types = ["CCC", "CCH", "HCH"],
        dihedral_types = ["CCCC", "HCCC", "HCCH"],
        pair_types = ["CCCC", "HCCC", "HCCH"],
        dtype = "double"
    )

    snapshot.particles.typeid[0] = 0
    snapshot.particles.typeid[1:4] = 1
    snapshot.particles.typeid[4] = 0
    snapshot.particles.typeid[5:7] = 1
    snapshot.particles.typeid[7] = 0
    snapshot.particles.typeid[8:10] = 1
    snapshot.particles.typeid[10] = 0
    snapshot.particles.typeid[11:14] = 1

    positions = numpy.array([
        [-2.990196,  0.097881,  0.000091],
        [-2.634894, -0.911406,  0.001002],
        [-2.632173,  0.601251, -0.873601],
        [-4.060195,  0.099327, -0.000736],
        [-2.476854,  0.823942,  1.257436],
        [-2.832157,  1.833228,  1.256526],
        [-2.834877,  0.320572,  2.131128],
        [-0.936856,  0.821861,  1.258628],
        [-0.578833,  1.325231,  0.384935],
        [-0.581553, -0.187426,  1.259538],
        [-0.423514,  1.547922,  2.515972],
        [-0.781537,  1.044552,  3.389664],
        [ 0.646485,  1.546476,  2.516800],
        [-0.778816,  2.557208,  2.515062]
    ])

    reference_box_low_coords = numpy.array([-22.206855, -19.677099, -19.241968])
    box_low_coords = numpy.array([
        -snapshot.box.Lx / 2,
        -snapshot.box.Ly / 2,
        -snapshot.box.Lz / 2
    ])
    positions += (box_low_coords - reference_box_low_coords)

    snapshot.particles.position[:] = positions[:]

    mC = 12.00
    mH = 1.008
    snapshot.particles.mass[:] = [
        mC, mH, mH, mH,
        mC, mH, mH,
        mC, mH, mH,
        mC, mH, mH, mH
    ]

    reference_charges = numpy.array([
        -0.180000, 0.060000, 0.060000, 0.060000,
        -0.120000, 0.060000, 0.060000,
        -0.120000, 0.060000, 0.060000,
        -0.180000, 0.060000, 0.060000, 0.060000]
    )
    charge_conversion = 18.22262
    snapshot.particles.charge[:] = charge_conversion * reference_charges[:]

    snapshot.bonds.resize(13)
    snapshot.bonds.typeid[0:3] = 1
    snapshot.bonds.typeid[3] = 0
    snapshot.bonds.typeid[4:6] = 1
    snapshot.bonds.typeid[6] = 0
    snapshot.bonds.typeid[7:9] = 1
    snapshot.bonds.typeid[9] = 0
    snapshot.bonds.typeid[10:13] = 1

    snapshot.bonds.group[:] = [
        [0, 2], [0, 1], [0, 3], [0, 4],
        [4, 5], [4, 6], [4, 7],
        [7, 8], [7, 9], [7, 10],
        [10, 11], [10, 12], [10, 13]
    ]

    snapshot.angles.resize(24)
    snapshot.angles.typeid[0:2] = 2
    snapshot.angles.typeid[2] = 1
    snapshot.angles.typeid[3] = 2
    snapshot.angles.typeid[4:8] = 1
    snapshot.angles.typeid[8] = 0
    snapshot.angles.typeid[9] = 2
    snapshot.angles.typeid[10:14] = 1
    snapshot.angles.typeid[14] = 0
    snapshot.angles.typeid[15] = 2
    snapshot.angles.typeid[16:21] = 1
    snapshot.angles.typeid[21:24] = 2

    snapshot.angles.group[:] = [
        [1, 0, 2], [2, 0, 3], [2, 0, 4],
        [1, 0, 3], [1, 0, 4], [3, 0, 4],
        [0, 4, 5], [0, 4, 6], [0, 4, 7],
        [5, 4, 6], [5, 4, 7], [6, 4, 7],
        [4, 7, 8], [4, 7, 9], [4, 7, 10],
        [8, 7, 9], [8, 7, 10], [9, 7, 10],
        [7, 10, 11], [7, 10, 12], [7, 10, 13],
        [11, 10, 12], [11, 10, 13], [12, 10, 13]
    ]

    snapshot.dihedrals.resize(27)
    snapshot.dihedrals.typeid[0:2] = 2
    snapshot.dihedrals.typeid[2] = 1
    snapshot.dihedrals.typeid[3:5] = 2
    snapshot.dihedrals.typeid[5] = 1
    snapshot.dihedrals.typeid[6:8] = 2
    snapshot.dihedrals.typeid[8:11] = 1
    snapshot.dihedrals.typeid[11] = 0
    snapshot.dihedrals.typeid[12:14] = 2
    snapshot.dihedrals.typeid[14] = 1
    snapshot.dihedrals.typeid[15:17] = 2
    snapshot.dihedrals.typeid[17:21] = 1
    snapshot.dihedrals.typeid[21:27] = 2

    snapshot.dihedrals.group[:] = [
        [2, 0, 4, 5], [2, 0, 4, 6], [2, 0, 4, 7],
        [1, 0, 4, 5], [1, 0, 4, 6], [1, 0, 4, 7],
        [3, 0, 4, 5], [3, 0, 4, 6], [3, 0, 4, 7],
        [0, 4, 7, 8], [0, 4, 7, 9], [0, 4, 7, 10],
        [5, 4, 7, 8], [5, 4, 7, 9], [5, 4, 7, 10],
        [6, 4, 7, 8], [6, 4, 7, 9], [6, 4, 7, 10],
        [4, 7, 10, 11], [4, 7, 10, 12], [4, 7, 10, 13],
        [8, 7, 10, 11], [8, 7, 10, 12], [8, 7, 10, 13],
        [9, 7, 10, 11], [9, 7, 10, 12], [9, 7, 10, 13]
    ]

    snapshot.pairs.resize(27)
    snapshot.pairs.typeid[0:1] = 0
    snapshot.pairs.typeid[1:11] = 1
    snapshot.pairs.typeid[11:27] = 2
    snapshot.pairs.group[:] = [
        # CCCC
        [0, 10],
        # HCCC
        [0, 8], [0, 9], [5, 10], [6, 10],
        [1, 7], [2, 7], [3, 7],
        [11, 4], [12, 4], [13, 4],
        # HCCH
        [1, 5], [1, 6], [2, 5], [2, 6], [3, 5], [3, 6],
        [5, 8], [6, 8], [5, 9], [6, 9],
        [8, 11], [8, 12], [8, 13], [9, 11], [9, 12], [9, 13]
    ]

    hoomd.init.read_snapshot(snapshot)

    ### Set interactions
    nl_ex = hoomd.md.nlist.cell()
    nl_ex.reset_exclusions(exclusions = ["1-2", "1-3", "1-4"])

    lj = hoomd.md.pair.lj(r_cut = 12.0, nlist = nl_ex)
    lj.pair_coeff.set('C', 'C', epsilon = 0.07, sigma = 3.55)
    lj.pair_coeff.set('H', 'H', epsilon = 0.03, sigma = 2.42)
    lj.pair_coeff.set('C', 'H', epsilon = numpy.sqrt(0.07*0.03), sigma = numpy.sqrt(3.55*2.42))

    coulomb = hoomd.md.charge.pppm(hoomd.group.charged(), nlist = nl_ex)
    coulomb.set_params(Nx = 64, Ny = 64, Nz = 64, order = 6, rcut = 12.0)

    harmonic = hoomd.md.bond.harmonic()
    harmonic.bond_coeff.set("CC", k = 2*268.0, r0 = 1.529)
    harmonic.bond_coeff.set("CH", k = 2*340.0, r0 = 1.09)

    angle = hoomd.md.angle.harmonic()
    angle.angle_coeff.set("CCC", k = 2*58.35, t0 = 112.7 * pi / 180)
    angle.angle_coeff.set("CCH", k = 2*37.5, t0 = 110.7 * pi / 180)
    angle.angle_coeff.set("HCH", k = 2*33.0, t0 = 107.8 * pi / 180)


    dihedral = hoomd.md.dihedral.opls()
    dihedral.dihedral_coeff.set("CCCC", k1 = 1.3, k2 = -0.05, k3 = 0.2, k4 = 0.0)
    dihedral.dihedral_coeff.set("HCCC", k1 = 0.0, k2 = 0.0, k3 = 0.3, k4 = 0.0)
    dihedral.dihedral_coeff.set("HCCH", k1 = 0.0, k2 = 0.0, k3 = 0.3, k4 = 0.0)

    lj_special_pairs = hoomd.md.special_pair.lj()
    lj_special_pairs.pair_coeff.set("CCCC", epsilon = 0.07, sigma = 3.55, r_cut = 12.0)
    lj_special_pairs.pair_coeff.set("HCCH", epsilon = 0.03, sigma = 2.42, r_cut = 12.0)
    lj_special_pairs.pair_coeff.set("HCCC",
        epsilon = numpy.sqrt(0.07 * 0.03), sigma = numpy.sqrt(3.55 * 2.42), r_cut = 12.0
    )

    coulomb_special_pairs = hoomd.md.special_pair.coulomb()
    coulomb_special_pairs.pair_coeff.set("CCCC", alpha = 0.5, r_cut = 12.0)
    coulomb_special_pairs.pair_coeff.set("HCCC", alpha = 0.5, r_cut = 12.0)
    coulomb_special_pairs.pair_coeff.set("HCCH", alpha = 0.5, r_cut = 12.0)

    hoomd.md.integrate.mode_standard(dt = dt)
    integrator = hoomd.md.integrate.nvt(group = hoomd.group.all(), kT = kT, tau = 100*dt)
    integrator.randomize_velocities(seed = 42)

    return hoomd.context.current
```

<!-- #region id="3UrzENm_oo6U" -->

Next, we load PySAGES and the relevant classes and methods for our problem

<!-- #endregion -->

```python id="fpMg-o8WomAA"
from pysages.grids import Grid
from pysages.colvars import DihedralAngle
from pysages.methods import FUNN

import pysages
```

<!-- #region id="LknkRvo1o4av" -->

The next step is to define the collective variable (CV). In this case, we choose the central dihedral angle.

We also define a grid to bin our CV space, the topology (tuple indicating the number of
nodes of each hidden layer) for our neural network which will model the free energy.

The appropriate number of bins depends on the complexity of the free energy landscape,
a good rule of thumb is to choose between 20 to 100 bins along each CV dimension
(using higher values for more rugged free energy surfaces), but it can be systematically
found trying different values for short runs of any given system.

<!-- #endregion -->

```python id="B1Z8FWz0o7u_"
cvs = [DihedralAngle([0, 4, 7, 10])]
grid = Grid(lower=(-pi,), upper=(pi,), shape=(64,), periodic=True)

topology = (14,)
method = FUNN(cvs, grid, topology)
```

<!-- #region id="Fz8BfU34pA_N" -->

We now simulate $5\times10^5$ time steps.
Make sure to run with GPU support, otherwise, it can take a very long time.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="K951m4BbpUar" outputId="f3e79872-41da-479a-caec-5bca7a6792e5"
run_result = pysages.run(method, generate_context, int(5e5))
```

<!-- #region id="PXBKUfK0p9T2" -->

Let's now plot the free energy landscape learned by the FUNN sampling method.
<!-- #endregion -->

```python id="X69d1R7OpW4P"
import matplotlib.pyplot as plt
```

```python id="6W7Xf0ilqAcm"
result = pysages.analyze(run_result)

mesh = result["mesh"]
A = result["free_energy"]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="TBiPAnMwqEIF" outputId="3a13a52d-2bd8-4122-db13-18bc6a11c797"
fig, ax = plt.subplots()

ax.set_xlabel(r"Dihedral Angle, $\xi$")
ax.set_ylabel(r"$\nabla A(\xi)$")

ax.plot(mesh, A)
plt.gca()
```

