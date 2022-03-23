# required modules
module unload gcc
module load gcc/10.2.0 cmake/3.15 eigen/3.3 cuda/11.2
 
# activate conda environment
source activate sd-pysages

python3 butane.py

