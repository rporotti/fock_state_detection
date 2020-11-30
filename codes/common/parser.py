import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser(description='Train a RL agent')

parser.add_argument('--T_max', type=float, default=1.0, help='Physical max time in ns (default 30 )')
parser.add_argument('--Nstates', type=int, default=15, help='Size of Hilbert space (default at 5)')
parser.add_argument('--meas', default="homodyne", help='homodyne or heterodyne')
parser.add_argument('--decay', type=float, default=0, help='Decay rate (default at 0)')
parser.add_argument('--meas_rate', type=float, default=25.0, help='Measurement rate (default at 0)')
parser.add_argument('--dephasing', type=float, default=0, help='Dephasing rate (default at 0)')
parser.add_argument('--N', type=int, default=8, help='Output channels')
parser.add_argument('--chi', type=int, default=1)

parser.add_argument('--ntraj', type=int, default=1, help='Number of RL env to run in parallel (default at 1)')
parser.add_argument('--obs', type=str, default="density_matrix", help='RL observation (density_matrix, measurement, time) (default at density_matrix)')
parser.add_argument('--timesteps', type=int, default=100, help='RL timesteps in an episode (default at 50)')
parser.add_argument('--substeps', type=int, default=10, help='RL substeps')
parser.add_argument('--lstm', action="store_true", help='Whethere use a LSTM or not (default at False)')
parser.add_argument('--tries', type=int, default=1, help='Number of RL env to run in parallel (default at 1)')
parser.add_argument('--fixed_seed', action="store_true", help='True to start from same seed each episode')
parser.add_argument('--multiplier', type=float, default=25.0, help='Multiplier for action')
parser.add_argument('--last_timesteps', type=int, default=10, help='Last timesteps for observation')
parser.add_argument('--filter', action="store_true", help='Filter the signal')
parser.add_argument('--discrete', action="store_true", help='Discrete actions')
parser.add_argument('--num_actions', type=int, default=2, help='Num of actions')
parser.add_argument('--capped_to_zero', action="store_true")
parser.add_argument('--size_filters', type=int, default=1)

parser.add_argument('--mode', type=str, default="script", help='Script or cluster (default at script)')
parser.add_argument('--library', type=str, default="SB", help='SB for stable_baselines, SU for spinning_up (default at stable_baselines)')
parser.add_argument('--power_reward', type=float, default=1, help='Default at 1')


parser.add_argument('--info', type=str, default="", help='additional info for the saving folder (default at None)')

parser.add_argument('--folder', type=str, default="", help='Saving folder (optional)')

parser.add_argument('--folder_policy', type=str, default="", help='Folder for the policy (if check_policy is True)')
parser.add_argument('--animation', action="store_true", help='To animate the simulation')
parser.add_argument('--simulation', action="store_true", help='To plot a simulation')
parser.add_argument('--compare_qutip', action="store_true",help='Compare with qutip')


parser.add_argument('--init_state', type=str, default="0", help='Inital Fock state')
parser.add_argument('--target_state', type=str, default="1", help='Target Fock state')
parser.add_argument('--same', action="store_true", help='Whether to stabilize the state')

parser.add_argument('--mpi', action="store_true", help='Use MPI on cluster')
parser.add_argument('--stop_best', action="store_true", help='Stop the training after best')
parser.add_argument('--stop_after', type=int, default=-1, help='Stop the training after this number of epochs')
parser.add_argument('--save_every', type=int, default=1, help='Save figure every save_every episodes')


parser.add_argument('--HER',dest="HER",action="store_true", help='Use HER')
parser.add_argument('--algorithm', type=str, default="PPO1", help='RL Algorithm (PPO1, PPO2, TRPO, DDPG, SAC)')


parser.add_argument('--timesteps_per_actorbatch', type=int, default=256)
parser.add_argument('--clip_param', type=float, default=0.2)
parser.add_argument('--entcoeff', type=float, default=0.01)
parser.add_argument('--optim_epochs', type=int, default=4)
parser.add_argument('--optim_stepsize', type=int, default=0.001)
parser.add_argument('--optim_batchsize', type=int, default=64)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lam', type=float, default=0.95)
parser.add_argument('--adam_epsilon', type=float, default=1e-05)

parser.add_argument('--RL_steps', type=float, default=1E6)



parser.add_argument('--main_folder', type=str, default="../simulations/", help='Main folder')

#args = parser.parse_args()

# if (args.check_policy):
#     assert args.folder_policy!="", "Must provide a folder for the policy!"

#
# if args.ntraj>1:
#     print("ntraj = "+str(args.ntraj)+ ": using PPO2 and multiprocessing")
# else:
#     print("ntraj = "+str(args.ntraj)+ ": using PPO1")
