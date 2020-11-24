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
import h5py
from mpi4py import MPI

import numpy as np


# n = 10

# num_processes = MPI.COMM_WORLD.size
# rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

# np.random.seed(746574366 + rank)

# f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
# f.create_dataset('reward', (num_processes, n), dtype='f')
# f.create_dataset('probs_fin', (num_processes, n), dtype='f')

# for j in range(n):
#    f["reward"][rank,j] = j+1
# f["probs_fin"][rank,0] = np.random.rand()

# f.close()




# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# if rank == 0:
#     data = "ciao"
#     data = comm.bcast(data, root=0)
# elif rank>0:
#     data=None
    
# data = comm.bcast(data, root=0)
# print(data)





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
    
    
    
class BaseEnv(gym.Env):

    def __init__(self,counter=0,q=None):
        super(BaselEnv, self).__init__()
        self.counter=counter
        self.queue=q
        
        if self.counter==0:
            self.queue.put("Hello")
        else:
            print(self.queue.get())
        #print(multiprocessing.current_process())
        time.sleep(0.1)



from stable_baselines.common.vec_env import SubprocVecEnv
import multiprocessing



def make_env(counter,q):
    def _init():
        env = BaseEnv(counter,q)
        return env
    return _init
    
    
if __name__ == '__main__':
    n_envs = 4
    q = multiprocessing.Queue()
    env = SubprocVecEnv([make_env(i, q) for i in range(n_envs)])
    