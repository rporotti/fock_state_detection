#! /bin/bash -l
#
# This file is a sample batch script for multi-threaded CPU applications (e.g. with pthread, OpenMP, ...).
#
# Standard output and error:
#SBATCH -o .logs/tjob_%A_%a_out.txt
#SBATCH -e .logs/tjob_%A_%a_err.txt
#
# Initial working directory:
#SBATCH -D ./
#
# Job Name:
#SBATCH -J multiple
#
# Queue (Partition):
#SBATCH --partition=standard
#
# Process management:
#SBATCH --array=0-375%100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8   # specify number of CPU cores (maximum: 32 on highfreq, 64 on highmem)
#
# Explicitly specify memory (default is maximum on node):
#SBATCH --mem=64GB
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
# Configure notification via mail:
#SBATCH --mail-type=none
#SBATCH --mail-user=<name>@mpl.mpg.de

# Load necessary modules here

module load cuda
module load anaconda

source activate mpi5




multiplier=($(seq 1 2 50))
meas_rate=($(seq 1 2 30))



count=0
for val3 in "${multiplier[@]}"; do
for val5 in "${meas_rate[@]}"; do
 #echo $count; echo $SLURM_ARRAY_TASK_ID
 if [ $count = $SLURM_ARRAY_TASK_ID ]
 then
   command="python script_training.py --init_state 0 --target_state 2 --T_max 1 --meas_rate $val5 --timesteps 100 --multiplier $val3 --obs density_matrix --substeps 10 --save_every 5 --ntraj 128 --num_actions 2 --Nstates 15 --folder hyp_search_06_11_2020"
   #command+=" $val1"
   $command
 fi
 count=$((count + 1))
done
done
done
done
done
