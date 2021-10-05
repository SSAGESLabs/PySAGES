FROM innocentbug/pysages-openmm

COPY . pysages
RUN cd pysages && pip install .