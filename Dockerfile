# trunk-ignore-all(hadolint/DL3006,hadolint/DL3042,hadolint/DL3013,hadolint/DL3059)
FROM ssages/pysages-openmm
WORKDIR /

RUN python -m pip install --upgrade pip
RUN python -m pip install ase gsd matplotlib "pyparsing<3"

# Install JAX and JAX-MD
RUN python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python -m pip install --upgrade jax-md jaxopt

COPY . /PySAGES
RUN pip install /PySAGES/
