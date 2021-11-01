FROM ssages/pysages-openmm

RUN apt-get -y install findutils

RUN pip freeze --user | cut -d'=' -f1 | xargs -n1 pip install -U
RUN python -m pip install gsd matplotlib

COPY . PySAGES
RUN python3 -m pip install --upgrade pip
RUN cd PySAGES && pip install .