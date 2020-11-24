#! /bin/bash -l
#
# This file is a sample batch script for multi-threaded CPU applications (e.g. with pthread, OpenMP, ...).
#
# Standard output and error:
#SBATCH -o .logs/tjob_%j_out.txt
#SBATCH -e .logs/tjob_%j_err.txt
#
# Initial working directory:
#SBATCH -D ./
#
# Job Name:
#SBATCH -J mpi
#
# Queue (Partition):
#SBATCH --partition=standard
#
# Process management:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1   # specify number of CPU cores (maximum: 32 on highfreq, 64 on highmem)
#
# Explicitly specify memory (default is maximum on node):
#SBATCH --mem=64GB
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
# Configure notification via mail:
#SBATCH --mail-type=none
#SBATCH --mail-user=<name>@mpl.mpg.de

# Load necessaru modules here

module load anaconda cuda

source activate conda_env

# Set number of CPUs per tasks for OpenMP programs
if [ ! -z $SLURM_CPUS_PER_TASK ] ; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
    export OMP_NUM_THREADS=1
fi

# Enable hyperthreading. Set it after modules load.
export SLURM_HINT=multithread

# Run the program
#srun python3 -m stable_baselines.ppo1.run_atari


srun python demo.py


#srun python script_training.py --init_state 0 --target_state 3 --T_max 10 --meas_rate 0.5 --timesteps 50 --multiplier 1 --obs diagonal --substeps 10 --save_every 1
