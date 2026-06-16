# trunk-ignore-all(hadolint/DL3006,hadolint/DL3042,hadolint/DL3013,hadolint/DL3059)
FROM ssages/pysages-openmm
WORKDIR /

RUN python -m pip install --upgrade pip
RUN python -m pip install ase dill "gsd<3.3" matplotlib "pyparsing<3"

# Install JAX and JAX-based libraries
# TODO: Remove jax pin once DLPack buffer alignment is fixed
RUN python -m pip install --upgrade "jax[cuda]==0.4.34" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python -m pip install --upgrade "jax==0.4.34" "dm-haiku<0.0.11" "e3nn-jax!=0.20.4" "jax-md>=0.2.7" jaxopt

COPY . /PySAGES
RUN pip install /PySAGES/

# Disable CAP_SYS_PTRACE requirement for vader's CMA single-copy path.
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
# Non-privileged setups (e.g. MPI CI jobs) can use `docker run --user pysages`.
RUN useradd -m -u 1000 pysages && chown -R pysages:pysages /PySAGES
