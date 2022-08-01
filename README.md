# Deep Reinforcement Learning for Quantum State Preparation with Weak Nonlinear Measurements

Supporting code for the [paper published on Quantum](https://quantum-journal.org/papers/q-2022-06-28-747/). 

Usage:

- Run sbatch mpi_sbatch.sh --target_state 2 for running training with MPI across 20 nodes
- Run sbatch PPO2.sh --target_state 2 --ntraj 256 for PPO2 training
- hyper_search.sh for hyperparameters search across parameters

For help on the parameters: python script_training.py --help



To save the conda env: conda env --name conda_env export > env.yml
To restore the conda env: conda env create -f=env.yml
