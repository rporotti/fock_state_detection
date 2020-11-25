import gym
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
gym.logger.set_level(40)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from codes.common.parser import parser
from codes.environment.multichannel import SimpleCavityEnv
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecCheckNan, VecNormalize
import matplotlib
matplotlib.rcParams['figure.dpi']=300

#args = parser.parse_args(args=[])
args = parser.parse_args()
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy


import stable_baselines
import psutil
import re

class PinnedSubprocVecEnv(stable_baselines.common.vec_env.SubprocVecEnv):
	
	def __init__(self, env_fns, *, start_method=None, cpu_cores=None):
		if cpu_cores is None:
			cpu_cores = available_cpu_cores()
		else:
			cpu_cores = [cpu_core.__index__() for cpu_core in cpu_cores]
		
		super().__init__(env_fns=env_fns, start_method=start_method)
		
		if len(cpu_cores) < len(self.processes):
			# or raise error?
			cpu_partitions = [[cpu_core] for cpu_core in (len(self.processes)//len(cpu_cores))*cpu_cores+cpu_cores[:len(self.processes)%len(cpu_cores)]]
		else:
			cpu_partitions = []
			start = 0
			for worker_idx, _ in enumerate(self.processes):
				end = start+len(cpu_cores)//len(self.processes)
				if worker_idx<len(cpu_cores)%len(self.processes):
					end += 1
				cpu_partitions.append(cpu_cores[start:end])
				start = end
		
		assert len(self.processes) == len(cpu_partitions)
		
		for worker_proc, cpu_partition in zip(self.processes, cpu_partitions):
			psutil.Process(worker_proc.pid).cpu_affinity(cpu_partition)


def available_cpu_cores():
	try:
		proc_status_file = open('/proc/%d/status'%psutil.Process().pid)
	except FileNotFoundError:
		raise OSError('system does not support procfs')
	else:
		for line in proc_status_file.readlines():
			match = re.search(
					r'^\s*Cpus_allowed_list\s*:(\s*[0-9]+(\s*\-\s*[0-9]+)?\s*(,\s*[0-9]+(\s*\-\s*[0-9]+)?\s*)?)$',
					line
			)
			
			if match:
				cpu_cores = []
				for part in match.group(1).split(','):
					part = [int(n) for n in part.split('-')]
					if len(part)==1:
						cpu_cores.extend(part)
					elif len(part)==2:
						a, b = part
						cpu_cores.extend(range(a, b+1))
					else:
						raise RuntimeError
				return cpu_cores
	
	raise RuntimeError


def make_env(args, d, q, i):
    def _init():
        env = SimpleCavityEnv(args, dataset=d, queue=q,counter=i)
        return env
    return _init


if __name__ == '__main__':
    if args.lstm == True:
        policy = MlpLstmPolicy
    else:
        policy = MlpPolicy

    if args.mpi == True:
        env = DummyVecEnv([lambda: SimpleCavityEnv(args)])
        # env=SimpleCavityEnv(args)
        model = stable_baselines.PPO1(policy, env, verbose=1)
    else:
        import functools
        import multiprocessing

        q = multiprocessing.Queue()
        dataset = multiprocessing.RawArray('d', args.ntraj*3)
        env = SubprocVecEnv([make_env(args, dataset, q, i) for i in range(args.ntraj)],start_method="fork")

        #env = SubprocVecEnv([functools.partial(SimpleCavityEnv, args, queue=q, counter=n) for n in range(args.ntraj)],
        #                    start_method="fork")
        # env=VecNormalize(env)
        model = stable_baselines.PPO2(policy, env, verbose=1, nminibatches=args.ntraj)

    from stable_baselines.common.callbacks import CheckpointCallback

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=env.get_attr("direc")[0] + "/model")

    model.learn(int(args.RL_steps),callback=checkpoint_callback)

