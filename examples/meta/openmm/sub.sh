module unload gcc
module load gcc/10.2.0 cuda/11.2 python/anaconda-2020.02

source activate sd-pysages-openmm

module unload python/anaconda-2020.02

python3 alanine-dipeptide-vac_openmm.py

