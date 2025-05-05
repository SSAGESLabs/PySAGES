---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="fb18f873" -->
<!-- trunk-ignore-all(markdownlint/MD001) -->
<!-- trunk-ignore-all(markdownlint/MD024) -->
<!-- #endregion -->
<!-- #region id="p49wJ0IjLAVD" -->

# Setting up the Environment

<!-- #endregion -->

<!-- #region id="0opWLiz1shLz" -->
We set up a directory that we will use as an installation prefix. If you are not running in an interactive environment and you don't want to install packages in a custom path, the steps in this section are unnecessary.

<!-- #endregion -->

```python id="YQplbeQbsvg_"
import os
import sys

ver = sys.version_info

os.environ["PYSAGES_ENV"] = os.environ["PREFIX"] = "/env/pysages"
os.environ["PYTHON_SITE_INSTALL_DIR"] = f"lib/python{str(ver.major)}.{str(ver.minor)}/site-packages"
os.environ["PREFIX_USER_SITE"] = os.environ["PREFIX"] + "/" + os.environ["PYTHON_SITE_INSTALL_DIR"]

# The following lines are to prevent python packages being looked up from certain paths in this Colab instance
for path in ("/content", ""):
  if path in sys.path:
    sys.path.remove(path)
```

```python id="f4YUpAAD_ECn"
!mkdir -p $PREFIX_USER_SITE
```

<!-- #region id="V3bkQyaIerAM" -->
We want to make the installation visible to the python system, but we will be installing packages into a custom location that is not checked by python by default. In this Colab, we achieve this in two steps.

First, we extend the environment variable `PYTHONPATH`. This helps newly started python environments to find packages.

<!-- #endregion -->

```python id="rK5eqf1Efd5U"
import os
os.environ["PYTHONPATH"] = os.environ["PREFIX_USER_SITE"] + ":" + os.environ["PYTHONPATH"]
```

<!-- #region id="WJpB7JxLflFa" -->
Because the notebook environment has already a running python we need to let it know about the new location. We achieve this by appending the `sys.path` with such location.
<!-- #endregion -->

```python id="UeVmb0cZfl8-"
import os
import sys
sys.path.append(os.environ["PREFIX_USER_SITE"])
```

<!-- #region id="S6jbsO0ZxUyO" -->

# Adding HOOMD-blue Support

We first install some dependencies necessary to build HOOMD-blue. These may vary in number and names based on your environment and operating system.
<!-- #endregion -->

```python id="tlM4nMxwXFKO"
!apt-get -qq install libeigen3-dev pybind11-dev > /dev/null
```

<!-- #region id="yreB95grry8d" -->

## Building and Installing HOOMD-blue

The following clones the HOOMD-blue repo and sets the version to `v4.7.0` (this is the newest version that builds with the system dependencies available in Ubuntu 22.04, which is the OS used in Colab as of the end of 2024).
<!-- #endregion -->

```bash id="bEvgKS5EZDW1"

# Get HOOMD-blue source code
rm -rf hoomd-blue
git clone -q https://github.com/glotzerlab/hoomd-blue.git
cd hoomd-blue
git checkout -q v4.7.0
git submodule update -q --init
```

<!-- #region id="Q-XrmUMJZF2N" -->
#### \_\_\_

We need to patch `CMake/hoomd/HOOMDPythonSetup.cmake` to being able to install
in a custom `site-packages` path within this Colab instance. This is also done
for hoomd conda builds (see for example [here](https://github.com/conda-forge/hoomd-feedstock/pull/106)).
In general you shouldn't need to do this.
<!-- #endregion -->

```bash id="kbfWJ0bGZsAt"
cd hoomd-blue
wget -q -O- https://raw.githubusercontent.com/conda-forge/hoomd-feedstock/4eb9b8ecd47f6780fcdbcde90ad99c180b5e2f51/recipe/fix-python-site-dir.patch | patch -p1 -s
```

<!-- #region id="pxQu1nVbc45X" -->
#### \_\_\_

We first disable some HOOMD-blue components to save on installation time, and then, we compile and install the package.

**This may take a while, so be mindful of not inadvertently re-running the cell.**
<!-- #endregion -->

```bash id="r_eiHnV5K6HI"

cd hoomd-blue

# Compile and install
BUILD_PATH=/tmp/build/hoomd
rm -rf $BUILD_PATH
cmake -S . -B $BUILD_PATH \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DBUILD_HPMC=OFF \
    -DBUILD_METAL=OFF \
    -DBUILD_MPCD=OFF \
    -DBUILD_TESTING=OFF \
    -DENABLE_GPU=ON \
    -DENABLE_TBB=ON \
    -DPLUGINS="" \
    -DPYTHON_SITE_INSTALL_DIR=$PYTHON_SITE_INSTALL_DIR/hoomd > /dev/null

cmake --build $BUILD_PATH --target install -j8 > /dev/null
# Or alternately to the line above
# cd $BUILD_PATH
# make install -j8 > /dev/null
```

<!-- #region id="jtPHo0j1aG5p" -->

## Building and Installing the HOOMD-dlext Plugin

Now we can install the `dlext` plugin for HOOMD-blue. But, we need to get some dependencies first.

<!-- #endregion -->

```python id="fAgQT1rRKsLf"
!python -m pip install -q setuptools_scm > /dev/null
```

<!-- #region id="13FChNadwLUn" -->

We then clone the hoomd-dlext repository and install the package via `cmake` as well. This cell is significantly faster than the HOOMD-blue installation.

<!-- #endregion -->

```bash id="-imFRwdKfDvq"

# Get the plugin
rm -rf hoomd-dlext
git clone -q https://github.com/SSAGESLabs/hoomd-dlext.git
cd hoomd-dlext

# Build and install
BUILD_PATH=/tmp/build/hoomd-dlext
rm -rf $BUILD_PATH
cmake -S . -B $BUILD_PATH -DCMAKE_FIND_ROOT_PATH=$PREFIX &> /dev/null
cmake --build $BUILD_PATH --target install > /dev/null
```

<!-- #region id="WVi8yFoDq--b" -->

This concludes the installation of the HOOMD-blue and its plugin for PySAGES. We quickly test the installation.

<!-- #endregion -->

```python id="xJC1ebpqrKC8"
import hoomd
import hoomd.dlext
```

<!-- #region id="gOGvNMRL2x3p" -->

# Adding OpenMM Support

Having previously set up the environment, we can now just simply install some required dependencies and build and install OpenMM.

Again, installing dependencies will be different depending on your operating system and python environment.
<!-- #endregion -->

```bash id="USDPtmzmBckY"

apt-get -qq install doxygen swig > /dev/null
python -m pip install -qq setuptools wheel Cython
```

<!-- #region id="Uaco_PJqoZrq" -->

## Building and Installing OpenMM

The following clones the OpenMM repo and sets the version to `v8.1.2` (the newest available when this notebook was last updated). Then, it configures and builds OpenMM.

**This may take a while, so be mindful of not inadvertently re-running the cell.**
<!-- #endregion -->

```bash id="OPsrb1RqmD-p"

# Get OpenMM source code
rm -rf openmm
git clone -q https://github.com/openmm/openmm.git
cd openmm
git checkout -q 8.1.2

# Compile and install
BUILD_PATH=/tmp/build/openmm
rm -rf $BUILD_PATH
cmake -S . -B $BUILD_PATH \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DBUILD_TESTING=OFF \
    -DOPENMM_PYTHON_USER_INSTALL=ON \
    -Wno-dev > /dev/null

cmake --build $BUILD_PATH -j8 &> /dev/null
cmake --install $BUILD_PATH > /dev/null
```

<!-- #region id="R8dVbeEdVyin" -->
The OpenMM python library needs to be build and installed separately. We also need to point for the library to be installed in our custom path.
<!-- #endregion -->

```bash id="WoEFi8P7XlVP"

export OPENMM_INCLUDE_PATH=$PREFIX/include
export OPENMM_LIB_PATH=$PREFIX/lib
BUILD_PATH=/tmp/build/openmm

# Install python package
cd $BUILD_PATH
make PythonInstall &> /dev/null

cd $BUILD_PATH/python
pip install --target $PREFIX_USER_SITE . &> /dev/null
```

<!-- #region id="zPZpzuaq9CIW" -->

## Building and Installing the OpenMM-dlext Plugin

Similarly as shown for HOOMD-blue above, for OpenMM we need to build and install the corresponding `openmm-dlext` plugin.

<!-- #endregion -->

```bash id="mCUYSTLp9M-C"

# Get the plugin
rm -rf openmm-dlext
git clone -q https://github.com/SSAGESLabs/openmm-dlext.git
cd openmm-dlext

# Build and install
BUILD_PATH=/tmp/build/openmm-dlext
rm -rf $BUILD_PATH
cmake -S . -B $BUILD_PATH -Wno-dev > /dev/null
cmake --build $BUILD_PATH --target install &> /dev/null
```

<!-- #region id="cm5xnNrM9P20" -->
If everything worked as expected, the following should run without issuing any errors.
<!-- #endregion -->

```python id="5Ty-Jnm09gnu"
import openmm
import openmm.dlext
```

<!-- #region id="5qvCIYnS3StP" -->

## Upload environment to Google Drive

<!-- #endregion -->

<!-- #region id="3a_zSXJatWUY" -->
These steps are not necessary to understand the setup of the environment. If you want to build your own environment, modify the lines such that it uploads to your own Google Drive.

We upload the data to a shared Google Drive so we can reuse our environment in other notebooks.

First, we mount our Google Drive file system to a local directory.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yic9Joq5tlGh" outputId="9ea110d5-8f45-4de2-e917-6c48d047acf9"
from google.colab import drive
drive.mount('/content/mnt')
```

<!-- #region id="lKHxKVDRYEpP" -->
We clean the cache of the code generated by python for our built packages such that the upload size is smaller.
<!-- #endregion -->

```bash id="Tl248P32YH8O"
python -m pip install -q pyclean > /dev/null
pyclean -q $PREFIX_USER_SITE
```

<!-- #region id="M09QOE_E3ukB" -->

We then compress the environment into a zip file and copy it to a folder within Google Drive. Here we are choosing an existing Shared Drive, but if you were to do this you should choose a folder you have access to and write permissions.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QmXbqp9Pqp-a" outputId="483ad6a2-2042-4a96-9eb1-964eb16c6c36"
%env PYSAGES_SHARED_ENV=/content/mnt/Shareddrives/pysages-env
```

```bash id="ufo9WHoQqzuW"

cd $PYSAGES_ENV
zip -qr pysages-env.zip .
cp -f pysages-env.zip $PYSAGES_SHARED_ENV
rm -f pysages-env.zip
```
