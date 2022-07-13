FROM ssages/pysages-openmm

RUN python -m pip install --upgrade pip
RUN python -m pip install ase gsd matplotlib "pyparsing<3"

# Install JAX
RUN python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY . PySAGES
RUN cd PySAGES && pip install .
