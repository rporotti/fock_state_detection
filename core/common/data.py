from common.parser import *
from functions.functions import split_string
from qutip import tensor, fock
import numpy as np

##### PHYSICAL DATA #######
args = parser.parse_args()



N_states=args.Nstates
measurement=args.meas

physical_model=args.physical_model

T_max=args.T_max
wc=args.wc
wq=args.wq
g=args.g

kappa=args.decay
kappa_meas=args.meas_rate
if measurement is not None and kappa_meas==0:
    print("Warning: measurement with rate==0!!")
gamma=args.qubit_decay
############################




##### RL DATA #######
ntraj=args.ntraj
bonus=False
obs=args.obs
timesteps=args.timesteps
substeps=args.substeps
if args.lstm==False:
    policy="MlpPolicy"
else:
    policy="MlpLstmPolicy"
lim_actions=[[-1,0],[-1,1]]
power_reward=args.power_reward

library=args.library
############################

mode=args.mode
stop_best=args.stop_best
stop_after=args.stop_after
additional_info=args.info
folder=args.folder
animation=args.animation
compare_qutip=args.compare_qutip
simulation=args.simulation
folder_policy=args.folder_policy

timesteps_per_actorbatch=args.timesteps_per_actorbatch
clip_param=args.clip_param
entcoeff=args.entcoeff
optim_epochs=args.optim_epochs
optim_stepsize=args.optim_stepsize
optim_batchsize=args.optim_batchsize
gamma=args.gamma
lam=args.lam
adam_epsilon=args.adam_epsilon

algorithm=args.algorithm
HER=args.HER

rho_init=args.init_state
rho_target=args.target_state
