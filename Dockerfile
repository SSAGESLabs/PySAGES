FROM ssages/pysages-openmm

RUN python -m pip install gsd matplotlib

COPY . PySAGES
RUN cd PySAGES && pip install .