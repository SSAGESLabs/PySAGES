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

<!-- trunk-ignore-all(markdownlint/MD001) -->
<!-- trunk-ignore-all(markdownlint/MD024) -->
<!-- #region id="p49wJ0IjLAVD" -->

# Installation of the environment

<!-- #endregion -->

<!-- #region id="WM_9PpDwKuoA" -->

First, we are setting up our environment, by installing HOOMD-blue and installing the dlext plugin for it. The first step is to update cmake to a version that is compatible with our installation process. In this case, we are using a newer version of CMake that is required by the HOOMD-dlext plugin.
This is special to the underlying (older) Ubuntu system of this Colab (18.04) and should not be required for newer installations.

<!-- #endregion -->

<!-- #region id="Bj0t7S1rcPoT" -->

# Update CMake

<!-- #endregion -->

```bash id="anZqRa7pxGpK"

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2> /dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg > /dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | tee /etc/apt/sources.list.d/kitware.list > /dev/null
apt-get -qq update
apt-get -qq remove cmake > /dev/null
apt-get -qq install cmake > /dev/null

# The following lines are specific to Colab installation environment
[[ -f /usr/local/bin/cmake && ! -f /usr/local/bin/cmake-default ]] && mv /usr/local/bin/cmake /usr/local/bin/cmake-default
ln -sf /usr/bin/cmake /usr/local/bin/cmake
```

<!-- #region id="0opWLiz1shLz" -->

##### \_\_\_

We set up a directory that we will use as an installation prefix (this can be different depending on where you want to install everything).

<!-- #endregion -->

```python id="YQplbeQbsvg_"
import os
import sys

ver = sys.version_info

os.environ["PYSAGES_ENV"] = "/env/pysages"
os.environ["PREFIX"] = "/env/pysages/lib/python" + str(ver.major) + "." + str(ver.minor) + "/site-packages"
```

```python id="f4YUpAAD_ECn"
!mkdir -p $PREFIX
```

<!-- #region id="S6jbsO0ZxUyO" -->

## HOOMD-blue

With this dependency updated, we can install HOOMD-blue.
The following cell clones and compiles HOOMD-blue manually.
We also make use of disabling components of the HOOMD-blue installation to save some installation time.

**This may take some time, be mindful of not inadvertently re-running the cell.**

<!-- #endregion -->

<!-- #region id="yreB95grry8d" -->

# Build and install HOOMD-blue

<!-- #endregion -->

```bash id="r_eiHnV5K6HI"

# Get HOOMD-blue source code
rm -rf hoomd-blue
git clone https://github.com/glotzerlab/hoomd-blue.git &>/dev/null
cd hoomd-blue
git checkout v2.9.7 &> /dev/null

# Compile and install
BUILD_PATH=/tmp/build/hoomd
rm -rf $BUILD_PATH
cmake -S . -B $BUILD_PATH \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DENABLE_CUDA=ON \
    -DBUILD_CGCMM=OFF \
    -DBUILD_DEM=OFF \
    -DBUILD_DEPRECATED=OFF \
    -DBUILD_HPMC=OFF \
    -DBUILD_METAL=OFF \
    -DBUILD_MPCD=OFF \
    -DBUILD_TESTING=OFF &> /dev/null

cmake --build $BUILD_PATH --target install -j8 &> /dev/null
# Or alternately to the line above
# cd /tmp/build/hoomd
# make install -j8 &> /dev/null
```

<!-- #region id="8ZxdRrN1rrtU" -->

##### \_\_\_

After the successful installation of HOOMD-blue, we make the installation visible to the python system. We installed HOOMD-blue into a custom location that is not checked by python for software out of the box.
In this Colab, this happens in two steps.

First, we extend the environment variable `PYTHONPATH`. This helps newly started python environments to find the HOOMD-blue installation. This is important for the installation of the plugin later. In none-notebook environments, this is the only step necessary.

<!-- #endregion -->

```python id="MuDzJEpFzev3"
import os
os.environ["PYTHONPATH"] = os.environ["PREFIX"] + ":" + os.environ["PYTHONPATH"]
```

<!-- #region id="fQkzEJWgzhCp" -->

##### \_\_\_

Because the notebook environment has already a running python we need to let this one know about the new package location. We achieve this by appending the `sys.path` with the location of our package.

<!-- #endregion -->

```python id="he413SCN-qKb"
import os
import sys
sys.path.append(os.environ["PREFIX"])
```

<!-- #region id="jtPHo0j1aG5p" -->

## HOOMD-dlext plugin

Now we can install the `dlext` plugin for HOOMD-blue. So we clone the hoomd-dlext repository and install the package via `cmake` as well.
This cell is significantly faster than the HOOMD-blue installation.

<!-- #endregion -->

<!-- #region id="13FChNadwLUn" -->

##### Build and install HOOMD-dlext plugin

<!-- #endregion -->

```bash id="-imFRwdKfDvq"

# Get the plugin
rm -rf hoomd-dlext
git clone https://github.com/SSAGESLabs/hoomd-dlext.git &> /dev/null
cd hoomd-dlext

# Build and install
BUILD_PATH=/tmp/build/hoomd-dlext
rm -rf $BUILD_PATH
cmake -S . -B $BUILD_PATH &> /dev/null
cmake --build $BUILD_PATH --target install > /dev/null
```

<!-- #region id="WVi8yFoDq--b" -->

##### \_\_\_

This concludes the installation of the HOOMD-blue and its Plugin for PySAGES. We quickly test the installation and, if successful, we proceed to also build and install OpenMM.

<!-- #endregion -->

```python id="xJC1ebpqrKC8"
import hoomd
import hoomd.dlext
```

<!-- #region id="gOGvNMRL2x3p" -->

## OpenMM

Having previously set up the environment variables for the HOOMD-blue installation, we can now just simply install some required dependencies and build and install OpenMM.

<!-- #endregion -->

<!-- #region id="t8d3toizoQe9" -->

##### \_\_\_

Installing dependencies will be different depending on your operating system and python environment.

<!-- #endregion -->

```bash id="USDPtmzmBckY"

apt-get -qq install doxygen swig > /dev/null
python -m pip install -q --upgrade pip setuptools wheel Cython &> /dev/null
```

<!-- #region id="Uaco_PJqoZrq" -->

##### Build and install OpenMM

<!-- #endregion -->

```bash id="OLmZh8mF8QQx"

# Get OpenMM source code
rm -rf openmm
git clone https://github.com/openmm/openmm.git &> /dev/null
cd openmm && git checkout 7.6.0 &> /dev/null

# Compile and install
BUILD_PATH=/tmp/build/openmm
rm -rf $BUILD_PATH
cmake -S . -B $BUILD_PATH \
    -DCMAKE_INSTALL_PREFIX=$PYSAGES_ENV \
    -DBUILD_TESTING=OFF \
    -Wno-dev &> /dev/null

cmake --build $BUILD_PATH -j8 &> /dev/null
cmake --install $BUILD_PATH &> /dev/null

# Install python package
export OPENMM_INCLUDE_PATH=$PYSAGES_ENV/include
export OPENMM_LIB_PATH=$PYSAGES_ENV/lib
cd $BUILD_PATH/python
pip install --target $PREFIX . &> /dev/null
```

<!-- #region id="zPZpzuaq9CIW" -->

## OpenMM-dlext plugin

Similarly we build and install the corresponding `openmm_dlext` plugin.

<!-- #endregion -->

<!-- #region id="N87ZPlxqZ-eS" -->

##### Build and install openmm-dlext

<!-- #endregion -->

```bash id="mCUYSTLp9M-C"

# Get the plugin
rm -rf openmm-dlext
git clone https://github.com/SSAGESLabs/openmm-dlext.git &> /dev/null
cd openmm-dlext

# Build and install
BUILD_PATH=/tmp/build/openmm-dlext
rm -rf $BUILD_PATH
cmake -S . -B $BUILD_PATH -Wno-dev &> /dev/null
cmake --build $BUILD_PATH --target install > /dev/null
```

<!-- #region id="cm5xnNrM9P20" -->

##### \_\_\_

Again, we test the installation and, if successful, we proceed to copy the environment to Google Drive to avoid building everything again in the future.

<!-- #endregion -->

```python id="5Ty-Jnm09gnu"
import openmm
import openmm_dlext
```

<!-- #region id="5qvCIYnS3StP" -->

## Upload environment to Google Drive

<!-- #endregion -->

<!-- #region id="3a_zSXJatWUY" -->

**This step can only be successfully executed by PySAGES maintainers.**

These steps are not necessary to understand the setup of the environment. If you want to build your own environment, modify the lines such that it uploads to your own Google Drive.

We upload the data to the shared Google Drive. First, we mount our Google Drive file system to a local directory.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yic9Joq5tlGh" outputId="6a516f55-ec92-4228-b0bb-f5f0fe9b43ec"
from google.colab import drive
drive.mount('/content/mnt')
```

<!-- #region id="M09QOE_E3ukB" -->

We then compress the environment into a zip file and copy it to a folder within Google Drive. Here we are choosing an existing Shared Drive, but if you were to do this you should choose a folder you have access to and write permissions.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QmXbqp9Pqp-a" outputId="046b3c5b-1980-43d4-bd6d-91d5d1b211a1"
%env PYSAGES_SHARED_ENV=/content/mnt/Shareddrives/pysages-env
```

```bash id="ufo9WHoQqzuW"

cd $PYSAGES_ENV
zip -qr pysages-env.zip .
cp -f pysages-env.zip $PYSAGES_SHARED_ENV
rm -f pysages-env.zip
```
