#!/bin/sh

#SBATCH --job-name=umbrella
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=01:00:00
#SBATCH --output=out.txt
#SBATCH --gres=gpu:4
#SBATCH --mem=20G

#python harmonic_bias.py
mpirun -n 5 python -m mpi4py.futures integration.py
