import gym
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
gym.logger.set_level(40)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import stable_baselines
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

from codes.common.parser import parser
from codes.environment.multichannel import SimpleCavityEnv
from codes.environment.pinnedsubprocvecenv import PinnedSubprocVecEnv


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
        if args.algorithm=="DDPG": policy=stable_baselines.ddpg.policies.MlpPolicy
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
        #env = SubprocVecEnv([functools.partial(SimpleCavityEnv, args, queue=q, counter=n) for n in range(args.ntraj)],
        #                    start_method="fork")
        # env=VecNormalize(env)
    if args.algorithm=="PPO2":
        model = stable_baselines.PPO2(policy, env, verbose=1, nminibatches=args.ntraj)
    else:
        model = getattr(stable_baselines, args.algorithm)(policy, env, verbose=1)

    from stable_baselines.common.callbacks import CheckpointCallback

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=env.get_attr("direc")[0] + "/model")

    model.learn(int(args.RL_steps),callback=checkpoint_callback)

