FROM innocentbug/pysages-openmm

COPY . PySAGES
RUN cd PySAGES && pip install .