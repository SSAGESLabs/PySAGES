FROM innocentbug/pysages-openmm

COPY . PySAGES
RUN cd pysages && pip install .