FROM ssages/pysages-openmm

RUN python -m pip install gsd matplotlib "pyparsing<3"

COPY . PySAGES
RUN cd PySAGES && pip install .
