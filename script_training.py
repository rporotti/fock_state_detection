import sys
import numpy as np
from core.common.parser import parser
from core.environment.multichannel import SimpleCavityEnv
from core.environment.pinnedsubprocvecenv import PinnedSubprocVecEnv

import stable_baselines
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy




#args = parser.parse_args(args=[])
args = parser.parse_args()


def make_env(args, d, q, i):
    def _init():
        env = SimpleCavityEnv(args, dataset=d, queue=q,counter=i)
        return env
    return _init


if __name__ == '__main__':
    if args.lstm == True:
        policy = MlpLstmPolicy
    else:
        if args.algorithm=="DDPG": 
            policy=stable_baselines.ddpg.policies.MlpPolicy
            param_noise = None
            action_noise = stable_baselines.common.noise.OrnsteinUhlenbeckActionNoise(mean=np.zeros(args.num_actions), sigma=float(0.1) * np.ones(args.num_actions))

        else: policy = MlpPolicy


    if args.mpi == True:
        env = DummyVecEnv([lambda: SimpleCavityEnv(args)])
    else:
        import functools
        import multiprocessing

        q = multiprocessing.Queue()
        dataset = multiprocessing.RawArray('d', args.ntraj*3)
        if sys.platform=="linux":
            env = PinnedSubprocVecEnv([make_env(args, dataset, q, i) for i in range(args.ntraj)],start_method="fork")
        if sys.platform == "darwin":
            env = SubprocVecEnv([make_env(args, dataset, q, i) for i in range(args.ntraj)], start_method="fork")

    if args.algorithm=="PPO2":
        model = stable_baselines.PPO2(policy, env, verbose=1,n_steps=args.n_steps,
                                      nminibatches=args.nminibatches)
    elif args.algorithm=="DDPG":
        model = stable_baselines.DDPG(policy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
    elif args.algorithm == "PPO1":
        model = stable_baselines.PPO1(policy, env, verbose=1,gamma=args.gamma,
                                      timesteps_per_actorbatch=args.timesteps_per_actorbatch,
                                      clip_param=args.clip_param, entcoeff=args.entcoeff,
                                      optim_epochs=args.optim_epochs,
                                      optim_stepsize=args.optim_stepsize, optim_batchsize=args.optim_batchsize,
                                      lam=args.lam, adam_epsilon=args.adam_epsilon)
    else:
        model = getattr(stable_baselines, args.algorithm)(policy, env, verbose=1)


    if args.save_model:
        from stable_baselines.common.callbacks import CheckpointCallback
        callback = CheckpointCallback(save_freq=int(1E4), save_path=env.get_attr("direc")[0] + "/model")
    else:
        callback=None
    model.learn(int(args.RL_steps),callback=callback)

