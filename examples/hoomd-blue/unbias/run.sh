#!/bin/bash

python3 gen_gsd.py && \
    python3 unbias.py && \
    echo -e "\n\nUnbiased CVs printed"
