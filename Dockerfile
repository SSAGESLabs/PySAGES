FROM ubuntu:20.04

# Do not put anything confidential in this directory.
# It is going to be read by Dockerhub and produces a public images.

ENV DEBIAN_FRONTEND="noninteractive" TZ="Chicago/United States"
# Install system dependencies
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get update && apt-get install -y python3 cmake nvidia-cuda-toolkit git libopenmpi-dev
RUN apt-get update && apt-get install -y gcc-8 g++-8 python-is-python3 python3-pip

ENV CC=gcc-8
ENV CXX=g++-8

RUN python -m pip install --upgrade pip
RUN python -m pip install jaxlib


#HOOMD-blue dependency
RUN git clone https://github.com/glotzerlab/hoomd-blue.git && cd hoomd-blue && git checkout v2.9.7 && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/hoomd-install -DENABLE_TESTING=OFF -DENABLE_CUDA=ON -DENABLE_MPI=ON .. && make -j $(nproc) install
ENV PYTHONPATH=${PYTHONPATH}:/hoomd-install

#HOOMD-blue dlext plugin
RUN git clone https://github.com/SSAGESLabs/hoomd-dlext.git && cd hoomd-dlext && mkdir build && cd build && cmake .. && make install


RUN apt-get update && apt-get install -y doxygen swig nvidia-cuda-dev nvidia-cuda-toolkit python3-setuptools cython3
RUN git clone https://github.com/openmm/openmm.git &&  cd openmm &&  git checkout 7.5.0 &&  mkdir build && cd build && cmake -DPYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=../install -DBUILD_TESTING=OFF .. && make -j 6 install
ENV OPENMM_INCLUDE_PATH=/openmm/install/include
ENV OPENMM_LIB_PATH=/openmm/install/lib
RUN cd openmm/build/python && python3 setup.py install

RUN git clone https://github.com/SSAGESLabs/openmm-dlext.git && cd openmm-dlext && mkdir build && cd build && cmake .. && make install

RUN python -m pip install gsd

COPY . pysages
RUN cd pysages && pip install .