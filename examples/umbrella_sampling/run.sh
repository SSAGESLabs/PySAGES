#!/bin/bash

python3 gen_gsd.py && \
    python3 umbrella.py && \
    echo -e "\n\nBiased position histogram in hist.pdf"
