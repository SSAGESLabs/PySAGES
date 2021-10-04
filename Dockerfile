FROM innocentbug/pysages-plugin

COPY . pysages
RUN cd pysages && pip install .