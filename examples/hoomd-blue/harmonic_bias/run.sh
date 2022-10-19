#!/bin/bash

python3 gen_gsd.py &&
	python3 harmonic_bias.py &&
	echo -e "\n\nBiased position histogram in hist.pdf"
