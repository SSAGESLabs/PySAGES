# trunk-ignore-all(hadolint/DL3006,hadolint/DL3042,hadolint/DL3013,hadolint/DL3059)
FROM ssages/pysages-openmm
WORKDIR /PySAGES/.docker_build

RUN python -m pip install --upgrade pip
RUN python -m pip install ase gsd matplotlib "pyparsing<3"

# Install JAX
RUN python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY . ../
RUN pip install ..
