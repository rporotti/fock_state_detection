#! /bin/bash -l
#
# This file is a sample batch script for multi-threaded CPU applications (e.g. with pthread, OpenMP, ...).
#
# Standard output and error:
#SBATCH -o .logs/tjob_%A_out.txt
#SBATCH -e .logs/tjob_%A_err.txt
#
# Initial working directory:
#SBATCH -D ./
#
# Job Name:
#SBATCH -J PPO2
#
# Queue (Partition):
#SBATCH --partition=standard
#
# Process management:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32   # specify number of CPU cores (maximum: 32 on highfreq, 64 on highmem)
#
# Explicitly specify memory (default is maximum on node):
#SBATCH --mem=64GB
#
# Wall clock limit:
#SBATCH --time=167:59:59
#
# Configure notification via mail:
#SBATCH --mail-type=none
#SBATCH --mail-user=<name>@mpl.mpg.de

# Load necessary modules here

module load anaconda cuda

source activate conda_env

python script_training.py "$@"

