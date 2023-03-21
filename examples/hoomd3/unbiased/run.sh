#!/bin/bash

python3 gen_gsd.py &&
	python3 unbiased.py &&
	echo -e "\n\nUnbiased CVs printed"
