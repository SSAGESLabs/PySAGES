FROM ssages/pysages-openmm

RUN python -m pip install gsd matplotlib

COPY . PySAGES
RUN python3 -m pip install --upgrade pip
RUN cd PySAGES && pip install .