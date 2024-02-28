# trunk-ignore-all(hadolint/DL3006,hadolint/DL3042,hadolint/DL3013,hadolint/DL3059)
FROM ssages/pysages-openmm
WORKDIR /

RUN python -m pip install --upgrade pip
RUN python -m pip install ase dill gsd matplotlib "pyparsing<3"

# Install JAX and JAX-based libraries
RUN python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python -m pip install --upgrade "dm-haiku<0.0.11" "e3nn-jax!=0.20.4" "jax-md>=0.2.7" jaxopt

COPY . /PySAGES
RUN pip install /PySAGES/
