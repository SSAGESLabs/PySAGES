#!/bin/bash

python3 gen_gsd.py && \
    python3 umbrella.py && \
    python3 analyze_gsd.py umbrella.gsd && \
    echo -e "\n\nBiased position histogram in hist.pdf"
