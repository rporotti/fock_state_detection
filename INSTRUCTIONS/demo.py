"""
Creates an HDF5 file with a single dataset of shape (channels, n),
filled with random numbers.
Writing to the different channels (rows) is parallelized using MPI.
Usage:
  mpirun -np 8 python demo.py
Small shell script to run timings with different numbers of MPI processes:
  for np in 1 2 4 8 12 16 20 24 28 32; do
      echo -n "$np ";
      /usr/bin/time --format="%e" mpirun -np $np python demo.py;
  done
"""

import gym
base_model=gym.Env
import psutil
import multiprocessing









    
    
    
    

from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecCheckNan, VecNormalize
import stable_baselines

import os

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


    


# class SimpleCavityEnv(base_model):
#     def __init__(self,counter=0):
#         self.counter=counter
#         print(multiprocessing.current_process())
#         print(self.counter)
#         #self.init_var()
        
        

#     def init_var(self):
#         #print(multiprocessing.current_process())
#         q = multiprocessing.Queue()
#         if self.counter==0:
            
#             q.put("ciao")
#         else:
#             q = multiprocessing.Queue()
#             event = q.get(block=True, timeout=10)
#             print(event)
#         q.close()

# def main():
#     ntraj=4
#     from stable_baselines.common import set_global_seeds, make_vec_env
#     import functools
    
#     env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
#     env=SubprocVecEnv([lambda: SimpleCavityEnv(i)  for i in range(ntraj)])


# if __name__ == '__main__':
#     main()
    

    
import time
import h5py
    
    
    
class BaseEnv(gym.Env):

    def __init__(self,counter=0,q=None, arr=None,n_envs=1):
        super(BaseEnv, self).__init__()
        self.counter=counter
        self.queue=q
        self.arr=np.frombuffer(arr, dtype=np.float64).reshape(n_envs,3)
        
        if self.counter==0:
            self.dic="Hello"
            for _ in range(n_envs):
                self.queue.put("Hello")

            #self.hf.create_dataset('rewards', (n_envs, 1), chunks=True)

        else:
            self.dic=self.queue.get()

        print(self.counter, self.dic)
        self.func()
        while not np.all(self.arr[:,1]): True
        if self.counter==0:
            self.render()
        #print(multiprocessing.current_process())


    def func(self):
        if self.counter>5:
            self.arr[self.counter,0]=self.counter
        self.arr[self.counter, 1] = True

    def render(self):
        for i in range(10):
            print(self.arr[i,0])
        self.arr[:, 1] = False
from stable_baselines.common.vec_env import SubprocVecEnv
import multiprocessing



def make_env(counter,q, arr, n_envs):
    def _init():
        env = BaseEnv(counter,q, arr, n_envs)
        return env
    return _init
    
    
if __name__ == '__main__':
    n_envs = 10
    import numpy as np
    q = multiprocessing.Queue()
    # Wrap X as an numpy array so we can easily manipulates its data.
    dataset = multiprocessing.RawArray('d', n_envs*3)

    env = SubprocVecEnv([make_env(i, q, X_np, n_envs) for i in range(n_envs)],start_method="fork")
    