import matplotlib

matplotlib.use('Agg')
import gym

gym.logger.set_level(40)
import numpy as np
import qutip as qt
from codes.functions.functions import generate_state, create_dir, create_info, print_info
from scipy import linalg
import copy
import scipy
import matplotlib.pyplot as plt
from textwrap import wrap
import json
from mpi4py import MPI
import os
import itertools

base_model = gym.Env
import h5py


class SimpleCavityEnv(gym.Env):
    def __init__(self, *args, testing=False, viewer=True, dataset=None, queue=None, counter=0):
        super(SimpleCavityEnv, self).__init__()
        self.testing = testing
        self.counter = counter
        self.viewer = viewer

        self.count = 0

        if queue is not None: self.queue = queue
        self.init_var(args, dataset)
        self.init_operators()
        self.set_placeholders()
        self.set_RL()

        self.ep = 0
        self.epoch = 0

        self.best_reward = -np.inf
        self.mode_init = "fixed"
        self.mode_target = "fixed"
        self.draw = False
        self.steps = 0
        self.figure = None

        if self.counter == 0 and self.rank == 0 and self.viewer:
            self.create_figure()

    def set_placeholders(self):
        self.xwigner = np.linspace(-5, 5, 100)
        self.cavity = np.zeros(self.T)
        self.rewards = np.zeros(self.T)
        #self.Rho_int = np.zeros(self.T, dtype=object)
        self.commut = np.zeros(self.T)
        self.actions_plot = np.zeros((self.num_actions, self.T))
        self.overlap = np.zeros(self.T)
        self.probabilities = np.zeros((self.Nstates, self.T * self.numberPhysicsMicroSteps))
        self.phases = np.zeros((self.Nstates, self.T * self.numberPhysicsMicroSteps))
        self.observations = np.zeros((self.T, 2, self.N))
        self.observations2 = np.zeros((self.T, 2, self.N))
        if self.filter:
            self.observations_filters = np.zeros((self.T, self.size_filters, 2, self.N,))
        self.total_rewards = [0]
        self.probs_final = [0]
        self.total_success = []
        self.success = 0
        self.av_tries = np.zeros(self.save_every)
        self.integrals = np.zeros((self.T, 2, self.N))
        self.std_rewards = [0]
        self.std_probs_final= [0]
        self.probs_final = [0]

        self.fid_finals = [0]
        self.rews = []
        self.successes = []
        self.fidelities = np.zeros(self.T)

    def set_RL(self):

        if self.discrete:
            self.action_space = gym.spaces.Discrete(5)
        else:
            self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.num_actions,), dtype=np.float32)
        if self.obs == "measure":
            if self.meas_type == "homodyne":
                self.MsmtTrace = np.zeros((self.last_timesteps, 2, self.N))
                if self.filter:
                    self.observation_space = gym.spaces.Box(low=-30, high=30,
                                                            shape=(self.last_timesteps, self.size_filters, 2, self.N),
                                                            dtype=np.float32)
                else:
                    self.observation_space = gym.spaces.Box(low=-30, high=30,
                                                            shape=(self.last_timesteps, 2, self.N),
                                                            dtype=np.float32)
                # self.observation_space = gym.spaces.Box(low=-30, high=30, shape=(self.N,))
            if self.meas_type == "heterodyne":
                self.MsmtTrace = np.zeros((self.last_timesteps, 2, self.N))
                self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(self.last_timesteps, 2, self.N),
                                                        dtype=np.float32)
        if self.obs == "density_matrix":
            self.MsmtTrace = np.zeros((self.last_timesteps, 1, self.N))
            self.observation_space = gym.spaces.Box(low=-1., high=1., shape=(2 * self.N_obs_rho * self.N_obs_rho,),
                                                    dtype=np.float32)
        if self.obs == "diagonal":
            self.MsmtTrace = np.zeros((self.last_timesteps, 1, self.N))
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.Nstates,), dtype=np.float32)

    def init_operators(self):
        self.a = qt.destroy(self.Nstates)
        self.ad = qt.create(self.Nstates)


        if self.num_actions<=2:
            self.P = np.zeros((1, self.N, self.Nstates, self.Nstates))
            self.P[0 ,np.arange(self.N), np.arange(self.N),np.arange(self.N)]=self.chi
        else:
            if not self.continuos_meas_rate:
                self.P = np.zeros((2**self.N, self.N, self.Nstates, self.Nstates))
                self.combinations = np.array(list(itertools.product([0, 1], repeat=self.N)))
                self.P[:, np.arange(self.N), np.arange(self.N), np.arange(self.N)]=self.combinations*self.chi

        self.aOp = self.a.full()
        self.adOp = self.ad.full()
        self.adaOp = np.matmul(self.adOp, self.aOp)
        self.aadOp = np.matmul(self.aOp, self.adOp)
        self.adaOp_square = np.matmul(self.adaOp, self.adaOp)
        self.a_plus_ad_Op = self.aOp + self.adOp

        if not self.continuos_meas_rate:
            self.H0=np.zeros(( 2**self.N, self.Nstates, self.Nstates),dtype="complex")
            for number in range(len(self.P)):
                self.H0[number] =np.sum(np.matmul(self.adaOp, self.P[number]),axis=0)

    def init_var(self, args, dataset):
        rank = MPI.COMM_WORLD.Get_rank()
        self.rank = rank
        if isinstance(args[0], dict) is False:
            dic = vars(args[0])
        else:
            dic = args[0]

        self.folder = dic["folder"]
        self.mode = dic["mode"]
        self.main_folder = dic["main_folder"]

        self.mpi = dic["mpi"]
        if self.mpi == True:
            self.ntraj = MPI.COMM_WORLD.Get_size()
        else:
            self.ntraj = dic["ntraj"]

        self.dic = dic
        self.N = dic["N"]
        self.N_obs_rho = dic["N_obs_rho"]
        self.chi = dic["chi"]
        self.Nstates = dic["Nstates"]  # Hilbert space
        self.kappa = dic["decay"]  # non-observed decay
        self.kappa_meas = dic["meas_rate"]  # measured channel
        self.kappa_dephasing = dic["dephasing"]  # measured channel
        self.numberPhysicsMicroSteps = dic["substeps"]  # micro steps per action/RL step
        self.obs = dic["obs"]
        self.meas_type = dic["meas"]
        self.num_actions = dic["num_actions"]
        self.capped = dic["capped_to_zero"]
        self.size_filters = dic["size_filters"]
        self.power_reward = dic["power_reward"]
        psi_init = generate_state(self.Nstates, dic["init_state"])
        self.Rho_initial = psi_init.proj().full()
        psi_target = generate_state(self.Nstates, dic["target_state"])
        self.Rho_target = psi_target.proj().full()
        self.RL_steps = int(dic["RL_steps"])

        self.T = dic["timesteps"]
        if dic["max_displ"]:
            self.max_displ=dic["max_displ"]
            self.T_max = dic["T_max"]

            self.dt = 1 / (self.T * self.numberPhysicsMicroSteps)
            #self.max_displ_per_step = self.max_displ / self.T * self.numberPhysicsMicroSteps * self.numberPhysicsMicroSteps
            self.max_displ_per_step = self.max_displ * self.T_max

            # if self.kappa_meas > 0:
            #     self.T_max *= self.kappa_meas
            #print(self.max_displ_per_step, self.dt, self.T)
        else:
            self.T_max = dic["T_max"]
            self.dt = self.T_max / (self.T * self.numberPhysicsMicroSteps)
            self.multiplier = float(dic["multiplier"])
            self.max_displ_per_step = self.multiplier / (self.T * self.numberPhysicsMicroSteps)

        self.t_mean = dic["t_mean"]
        self.fixed_seed = dic["fixed_seed"]
        if self.kappa_meas>0:
            T_max=self.T_max * self.kappa_meas
        else:
            T_max = self.T_max
        self.tlist = np.linspace(0, T_max, self.T)
        self.tlist_mean = np.linspace(0, T_max, int(self.T / self.t_mean))
        self.last_timesteps = dic["last_timesteps"]
        self.filter = dic["filter"]
        self.discrete = dic["discrete"]
        self.save_every = dic["save_every"]
        self.scale = np.sqrt(self.kappa_meas)
        self.continuos_meas_rate=dic["continuos_meas_rate"]
        self.name_folder=dic["name_folder"]


        if not self.testing:
            if self.counter == 0 and self.rank == 0:
                direc = create_dir(dic)
                info = create_info(dic)
                self.info = info
                self.direc = direc

                os.makedirs(self.direc + "/script", exist_ok=True)
                os.makedirs(self.direc + "/model", exist_ok=True)
                if self.folder != "":
                    os.makedirs(self.direc + "/../summaries", exist_ok=True)
                    os.makedirs(self.direc+ "/../current", exist_ok=True)
                os.system("cp script_training.py " + self.direc + "/script")
                os.system("cp codes/environment/multichannel.py " + self.direc + "/script")
                print_info(dic, direc)

            if self.mpi:
                if rank == 0:
                    MPI.COMM_WORLD.bcast(direc, root=0)
                    MPI.COMM_WORLD.bcast(info, root=0)
                else:
                    self.direc = MPI.COMM_WORLD.bcast(None, root=0)
                    self.info = MPI.COMM_WORLD.bcast(None, root=0)
                self.hf = h5py.File(self.direc + '/data.hdf5', 'a', driver='mpio', comm=MPI.COMM_WORLD)
                self.hf.create_dataset('rewards', (MPI.COMM_WORLD.size, 1), chunks=True)
                self.hf.create_dataset('probs_fin', (MPI.COMM_WORLD.size, 1), chunks=True)
                #self.hf.create_dataset('density_matrix', (MPI.COMM_WORLD.size, 2 * self.Nstates * self.Nstates), chunks=True)
            else:
                if self.counter == 0:
                    for _ in range(self.ntraj):
                        self.queue.put(self.direc)

                else:
                    self.direc = self.queue.get()
                self.arr = np.frombuffer(dataset, dtype=np.float64).reshape(self.ntraj, 3)
                self.queue.close()

            # print(multiprocessing.current_process())


    #         if self.folder!="" and self.counter==0 and self.rank==0:
    #             self.summary = open("../simulations/"+self.folder+"/summary.txt","a",os.O_NONBLOCK)
    #             self.summary.write(self.info+"\t")
    #             self.summary.close()

    def reset(self):
        if self.mode_init == "random":
            alpha = 1 + np.random.rand()
            init_state = qt.coherent(self.Nstates, alpha)
            self.Rho = init_state.proj().full()

        if self.mode_init == "fixed":
            self.Rho = copy.deepcopy(self.Rho_initial)

        if self.mode_target == "same":
            self.RhoGoal = copy.deepcopy(self.Rho)
            self.sqrtRhoGoal = scipy.linalg.sqrtm(self.RhoGoal)
        if self.mode_target == "fixed":
            self.RhoGoal = copy.deepcopy(self.Rho_target)
            self.sqrtRhoGoal = scipy.linalg.sqrtm(self.RhoGoal)
        self.target_distrib = np.real(np.diag(self.Rho_target))
        self.integrals[0,0,:self.N]=self.target_distrib[:self.N] # TODO
        self.compute_fidelity()
        self.rew_init = self.fidelity
        self.msmt_trace_size = 1
        self.X = np.zeros((2, self.N))
        self.t = 0
        if self.filter:
            self.y = np.zeros((self.last_timesteps,self.size_filters, 2, self.N)) # TODO
            self.y[-1, 0,0, :self.N] = self.target_distrib[:self.N] # TODO

        if self.fixed_seed:
            np.random.seed(0)
            self.rand=np.random.RandomState(0)
        else:
            self.rand=np.random.RandomState()
        return self._get_obs()

    def step(self, action):
        self.steps += 1
        self.done = False
        self.MsmtTrace[0:-1] = self.MsmtTrace[1:]  # shift
        self.MsmtTrace[-1] = 0

        if np.any(np.isnan(action)):
            self.done = True

        if np.any(np.isnan(action)):
            action = np.zeros(self.num_actions)

        if self.discrete:
            action = self.max_displ_per_step * action / 5
        else:
            if self.capped:
                action[:2] *= self.max_displ_per_step
                action[:2]+=1
                action[:2]/=2
            else:
                action[:2] *= self.max_displ_per_step

            number=0

            if self.num_actions==1:
                alpha = action[0]
            if self.num_actions >= 2:
                alpha = (action[0] + 1j * action[1]) / np.sqrt(2)

            if self.num_actions<= 2:
                P = self.P[0]
                H0 = self.H0[0]
            else:
                if self.continuos_meas_rate:
                    action[2:]=(action[2:]+1)/2
                    P = np.zeros((self.N, self.Nstates, self.Nstates))
                    P[np.arange(self.N), np.arange(self.N), np.arange(self.N)] =1
                    H0 = np.sum(action[2:,None,None]* np.matmul(self.adaOp, P),axis=0)


                else:
                    action[2:]=np.ceil(np.array(action)[2:]).clip(min=0)
                    appo=np.pad(action[2:], (0,self.N+2-self.num_actions),mode="constant",constant_values=(None,1))

                    number=int("".join([str(int(i)) for i in appo]),2)
                    P=self.P[number]
                    H0=self.H0[number]

        
        H_displacement = -1j * (alpha * self.adOp - np.conj(alpha) * self.aOp) / 2
        H = np.add(H_displacement,H0)

        for step in range(self.numberPhysicsMicroSteps):

            k1 = -1j * (np.matmul(H, self.Rho) - np.matmul(self.Rho, H)) * self.dt
            k2 = -1j * (np.matmul(H, self.Rho + k1 / 2) - np.matmul(self.Rho + k1 / 2, H)) * self.dt
            k3 = -1j * (np.matmul(H, self.Rho + k2 / 2) - np.matmul(self.Rho + k2 / 2, H)) * self.dt
            k4 = -1j * (np.matmul(H, self.Rho + k3) - np.matmul(self.Rho + k3, H)) * self.dt
            unitary = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            # unitary=-1j*(np.matmul(H,self.Rho)-np.matmul(self.Rho,H))*self.dt

            decay = 0
            dephasing = 0
            if self.kappa > 0:
                k1 = self.dt * self.kappa * (np.matmul(self.aOp, np.matmul(self.Rho, self.adOp)) -
                                             0.5 * (np.matmul(self.adaOp, self.Rho) + np.matmul(self.Rho, self.adaOp)))
                k2 = self.dt * self.kappa * (np.matmul(self.aOp, np.matmul(self.Rho + k1 / 2, self.adOp)) -
                                             0.5 * (np.matmul(self.adaOp, self.Rho + k1 / 2) + np.matmul(
                            self.Rho + k1 / 2, self.adaOp)))
                k3 = self.dt * self.kappa * (np.matmul(self.aOp, np.matmul(self.Rho + k2 / 2, self.adOp)) -
                                             0.5 * (np.matmul(self.adaOp, self.Rho + k2 / 2) + np.matmul(
                            self.Rho + k2 / 2, self.adaOp)))
                k4 = self.dt * self.kappa * (np.matmul(self.aOp, np.matmul(self.Rho + k3, self.adOp)) -
                                             0.5 * (np.matmul(self.adaOp, self.Rho + k3) + np.matmul(self.Rho + k3,
                                                                                                     self.adaOp)))
                decay = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            if self.kappa_dephasing > 0:
                k1 = self.dt / 2 * self.kappa_dephasing * ( \
                            np.matmul(P, np.matmul(self.Rho, P)) -
                            0.5 * (np.matmul(P, self.Rho) + np.matmul(self.Rho, P)))
                k2 = self.dt / 2 * self.kappa_dephasing * ( \
                            np.matmul(P, np.matmul(self.Rho + k1 / 2, P)) -
                            0.5 * (np.matmul(P, self.Rho + k1 / 2) + np.matmul(self.Rho + k1 / 2, P)))
                k3 = self.dt / 2 * self.kappa_dephasing * ( \
                            np.matmul(P, np.matmul(self.Rho + k2 / 2, P)) -
                            0.5 * (np.matmul(P, self.Rho + k2 / 2) + np.matmul(self.Rho + k2 / 2, P)))
                k4 = self.dt / 2 * self.kappa_dephasing * ( \
                            np.matmul(P, np.matmul(self.Rho + k3 / 2, P)) -
                            0.5 * (np.matmul(P, self.Rho + k3 / 2) + np.matmul(self.Rho + k3 / 2, P)))
                dephasing = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            # k=np.zeros((self.N,self.Nstates,self.Nstates),dtype=np.complex128)

            k_homodyne = 0
            dissipator = 0
            if self.kappa_meas > 0.:
                k1 = self.dt / 2 * self.kappa_meas * ( \
                            np.matmul(P, np.matmul(self.Rho, P)) -
                            0.5 * (np.matmul(P, self.Rho) + np.matmul(self.Rho, P)))
                k2 = self.dt / 2 * self.kappa_meas * ( \
                            np.matmul(P, np.matmul(self.Rho + k1 / 2, P)) -
                            0.5 * (np.matmul(P, self.Rho + k1 / 2) + np.matmul(self.Rho + k1 / 2, P)))
                k3 = self.dt / 2 * self.kappa_meas * ( \
                            np.matmul(P, np.matmul(self.Rho + k2 / 2, P)) -
                            0.5 * (np.matmul(P, self.Rho + k2 / 2) + np.matmul(self.Rho + k2 / 2, P)))
                k4 = self.dt / 2 * self.kappa_meas * ( \
                            np.matmul(P, np.matmul(self.Rho + k3 / 2, P)) -
                            0.5 * (np.matmul(P, self.Rho + k3 / 2) + np.matmul(self.Rho + k3 / 2, P)))
                dissipator = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                dW = self.rand.randn(self.N) * np.sqrt(self.dt)
                temp = np.matmul(P, self.Rho) + np.matmul(self.Rho, P)
                quadrature = np.real(np.trace(temp, axis1=1, axis2=2))
                self.X[0] = np.sqrt(self.kappa_meas / 2) * quadrature + dW / self.dt

                k1 = np.sqrt(self.kappa_meas / 2) * dW[:, np.newaxis, np.newaxis] * (
                            temp - quadrature[:, np.newaxis, np.newaxis] * self.Rho)
                k2 = np.sqrt(self.kappa_meas / 2) * dW[:, np.newaxis, np.newaxis] * (
                            temp - quadrature[:, np.newaxis, np.newaxis] * self.Rho + k1 / 2)
                k3 = np.sqrt(self.kappa_meas / 2) * dW[:, np.newaxis, np.newaxis] * (
                            temp - quadrature[:, np.newaxis, np.newaxis] * self.Rho + k2 / 2)
                k4 = np.sqrt(self.kappa_meas / 2) * dW[:, np.newaxis, np.newaxis] * (
                            temp - quadrature[:, np.newaxis, np.newaxis] * self.Rho + k3)
                k_homodyne = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                if self.meas_type == "heterodyne":
                    k_heterodyne = 0
                    k_heterodyne += dissipator

                    dW = np.random.randn(self.N) * np.sqrt(self.dt)
                    temp = np.matmul(P, self.Rho) - np.matmul(self.Rho, P)
                    quadrature = np.real(np.trace(temp, axis1=1, axis2=2))
                    self.X[1] = np.sqrt(self.kappa_meas) * quadrature + dW / self.dt
                    k1 = -1j * np.sqrt(self.kappa_meas / 2) * dW[:, np.newaxis, np.newaxis] * (
                                temp - quadrature[:, np.newaxis, np.newaxis] * self.Rho)
                    k2 = -1j * np.sqrt(self.kappa_meas / 2) * dW[:, np.newaxis, np.newaxis] * (
                                temp - quadrature[:, np.newaxis, np.newaxis] * self.Rho + k1 / 2)
                    k3 = -1j * np.sqrt(self.kappa_meas / 2) * dW[:, np.newaxis, np.newaxis] * (
                                temp - quadrature[:, np.newaxis, np.newaxis] * self.Rho + k2 / 2)
                    k4 = -1j * np.sqrt(self.kappa_meas / 2) * dW[:, np.newaxis, np.newaxis] * (
                                temp - quadrature[:, np.newaxis, np.newaxis] * self.Rho + k3)
                    k_heterodyne += k1

            self.Rho += unitary + np.sum(k_homodyne + dissipator+dephasing, axis=0) + decay
            if self.meas_type == "heterodyne":
                self.Rho += np.sum(k_heterodyne, axis=0)


            self.probabilities[:, int(self.t * self.numberPhysicsMicroSteps + step)] = np.abs(np.diag(self.Rho))
            self.phases[:, int(self.t * self.numberPhysicsMicroSteps + step)] = np.angle(np.diag(self.Rho))

        self.actions_plot[:, self.t] = action

        # print(np.linalg.norm(self.probabilities[:,self.t]))
        if np.any(np.isnan(self.Rho)) or np.trace(self.Rho) > 1.1:
            self.reset()
        self.cavity[self.t] = abs(np.trace(np.matmul(self.adaOp, self.Rho)))
        #self.Rho_int[self.t] = self.Rho
        self.compute_fidelity()
        if np.any(np.isnan(self.Rho)) or self.fidelity > 1.1:
            self.reset()

          # add the new msmt result




        # print(self.t, self.dt,self.observations)

        obs = self._get_obs()
        reward = self.getReward()  #####-1/8*(  np.sqrt(action[0]**2+action[1]**2)  )

        if self.t >= self.T-1:
            self.done = True
            self.end_episode()
        self.t += 1

        return obs, reward, self.done, {}

    def _get_obs(self):
        self.observations[self.t] = self.X

        if self.meas_type == "homodyne":
            self.MsmtTrace[-1] = self.X[0]
            if self.scale > 0:
                self.observations[self.t] = self.observations[self.t]/self.scale
        if self.meas_type == "heterodyne":
            self.MsmtTrace[-1] = self.X
        if self.t>0: # TODO
            self.integrals[self.t] = self.integrals[self.t - 1]*(self.t)/(self.t+1)+self.observations[self.t]/self.t # TODO


        if self.obs == "measure":
            if self.meas_type == "homodyne":

                if self.filter: # TODO
                    self.y[0:-1] = self.y[1:]  # shift
                    self.y[-1] = 0
                    taos = [50]
                    for i in range(self.size_filters):
                        self.y[-1,i] += 1 / taos[i] * (self.observations[self.t, 0] - self.y[-2,i])
                    obs = self.y
                    # obs=scipy.signal.savgol_filter(self.observations[:self.t,0],9,1,axis=0,mode="constant")[-self.last_timesteps]
                    print(obs.shape)
                    self.observations_filters[self.t ] = obs[-1]
                    return np.array(obs)

                else:
                    return np.array(self.MsmtTrace)/self.scale

            if self.meas_type == "heterodyne":
                return np.array(self.MsmtTrace)

        if self.obs == "time":
            return np.array(self.t)
        if self.obs == "density_matrix":
            a = self.Rho[:self.N_obs_rho,:self.N_obs_rho].flatten()
            obs = np.concatenate((np.real(a), np.imag(a)), axis=None)

            return obs
        if self.obs == "diagonal":
            obs = np.real(np.diag(self.Rho))

            return obs

    def compute_fidelity(self):

        self.fidelity = \
            abs(np.trace(scipy.linalg.sqrtm(np.matmul(np.matmul(self.sqrtRhoGoal, self.Rho), self.sqrtRhoGoal)))) ** 2

        if np.trace(self.Rho) > 1.01:
            # self.done=True
            self.fidelity = -1
            print("Fidelity > 1")

    def getReward(self):
        # self.compute_fidelity()
        rew =((    self.fidelity   ) ** self.power_reward    )/self.T
        #self.rew_init
        rew_not_normalized = (self.fidelity) ** self.power_reward


        # p=np.abs(np.real(self.probabilities[:,self.t-1]))
        # #print(p)
        # q=self.target_distrib
        # p = np.asarray(p, dtype=np.float)
        # q = np.asarray(q, dtype=np.float)
        # rew=-scipy.spatial.distance.jensenshannon(q,p)

        # print(rew)
        # if self.t==self.T*self.numberPhysicsMicroSteps:
        #     rew=self.fidelity
        # else:
        #     rew=0
        self.fidelities[self.t] = self.fidelity
        self.rewards[self.t] = rew


        return rew_not_normalized

    def set_rho_init(self, init_state):
        self.Rho_initial = init_state.full()

    def set_rho_target(self, target_state):
        self.Rho_target = target_state.full()

    def end_episode(self):
        # if self.rank == 0 and self.counter == 0:
        #
        #     self.total_rewards.append(np.sum(self.rews[-1]))
        #     self.total_success.append(np.mean(self.success))
        #
        #     # self.std_rewards.append(np.std(self.rews[-self.save_every:]))
        #     if self.viewer == True and self.ep % self.save_every == 0:
        #         print("Evaluating policy...")
        #         print("Last " + str(self.save_every) + " episodes, av. reward=" + str(
        #             np.mean(self.total_rewards[-self.save_every:])))
        #         print("Last " + str(self.save_every) + " episodes, prob. success=" + str(
        #             np.mean(self.total_rewards[-self.save_every:])))
        #         self.render()
        #self.probs_final.append(self.fidelity)
        self.rews.append(np.sum(self.rewards))

        if not self.testing:
            if self.mpi:
                self.hf["rewards"][self.rank] = np.sum(self.rewards)
                self.hf["probs_fin"][self.rank] = self.fidelity
                #self.hf["density_matrix"][self.rank] = self._get_obs()
            else:
                self.arr[self.counter, 0] = np.sum(self.rewards)
                self.arr[self.counter, 1] = self.fidelity

        if self.testing == False and self.rank == 0 and self.counter == 0:
            if self.mpi:
                total_rewards = np.mean(self.hf["rewards"][:, 0], axis=0)
                std_rewards = np.std(self.hf["rewards"][:, 0], axis=0)
                probs_fin = np.mean(self.hf["probs_fin"][:, 0], axis=0)
                std_probs_fin = np.std(self.hf["probs_fin"][:, 0], axis=0)
                #dens_mat_sum=np.mean(self.hf["density_matrix"], axis=0)
                #dens_mat=dens_mat_sum[:self.Nstates*self.Nstates]+1j*dens_mat_sum[-self.Nstates * self.Nstates:]
                #dens_mat=dens_mat.reshape(self.Nstates,self.Nstates)
                #fid_final=abs(np.trace(scipy.linalg.sqrtm(np.matmul(np.matmul(self.sqrtRhoGoal, dens_mat), self.sqrtRhoGoal)))) ** 2

            else:
                total_rewards = np.mean(self.arr[:, 0])
                std_rewards = np.std(self.arr[:, 0])
                probs_fin = np.mean(self.arr[:, 1])
                std_probs_fin = np.std(self.arr[:, 1])
            self.total_rewards.append(total_rewards)
            self.std_rewards.append(std_rewards)
            self.probs_final.append(probs_fin)
            self.std_probs_final.append(std_probs_fin)

            #self.fid_finals.append(fid_final)
            if self.ep % self.save_every == 0:
                self.render()
        self.ep += 1

    def create_figure(self):

        #
        lw = 1

        dpi = 150
        plt.rcParams.update({'font.size': 8})
        plt.rcParams.update({'figure.dpi': dpi})
        self.figure = plt.figure(figsize=(8, 5), constrained_layout=True)
        self.axes_obs = np.zeros((2, 4), dtype="object")
        #self.axes_integral = np.zeros((2, 4), dtype="object")

        gs = self.figure.add_gridspec(3, 8)
        offset = 0


        self.ax_trace = self.figure.add_subplot(gs[offset, :-2])
        self.ax_histo = self.figure.add_subplot(gs[offset, -2])

        self.ax_histo.get_xaxis().set_visible(False)
        self.ax_histo.get_yaxis().set_visible(False)
        self.ax_histo.set_xlim(0, 1)
        self.ax_histo.set_ylim(0, self.Nstates)

        self.axes_actions = self.figure.add_subplot(gs[1 + offset, :-2])

        if not self.testing:
            self.ax_reward = self.figure.add_subplot(gs[-1, -2:])

            self.ax_reward.set_xlabel('Trajectories')
            self.ax_reward.set_ylabel("Reward")
            self.ax_reward.set_ylim(0, 1)


            #self.ax_histo2.get_xaxis().set_visible(False)
            #self.ax_histo_cumulative.get_yaxis().set_visible(False)
            #self.ax_histo_cumulative.get_xaxis().set_ticks(np.linspace(0, 1, 5))
            #self.ax_histo_cumulative.set_xticklabels([])
            #self.ax_histo_cumulative.set_ylabel("# cases")
            #self.ax_histo_cumulative.set_ylabel("Fidelity")
            self.ax_histo_cumulative = self.figure.add_subplot(gs[1 + offset, -2:])
            self.ax_histo_cumulative.set_ylim(0, 1)
            self.ax_histo_cumulative.set_ylabel("Final fidelity")

        # self.ax_trace.yaxis.set_ticks_position('both')
        self.ax_trace.tick_params(axis='y', which='both', labelleft='on', labelright='on')
        self.ax_trace.yaxis.set_ticks_position('both')
        self.ax_trace.set_ylabel(r"$<a^{\dagger} a>$", labelpad=0);
        # self.ax_trace.set_yticks(np.arange(0,self.Nstates+1,2))
        labels = list(np.arange(self.Nstates))
        for i in np.arange(1, self.Nstates, 2):
            labels[i] = ""

        self.ax_trace.yaxis.set(ticks=np.arange(0.5, self.Nstates), ticklabels=labels)

        # self.ax_trace.yaxis.set_major_locator(MaxNLocator(integer=True))
        # self.axes[self.offset,1].set_ylabel(r"$<\sigma_m^{\dagger} \sigma_m>$", labelpad=0);
        # self.axes[self.offset,2].set_ylabel("Overlap, purity", labelpad=0);
        # self.axes[self.offset,3].set_ylabel("Measurement");


        appo = np.full(self.T, None)
        x = np.linspace(0, self.ep * self.ntraj, len(self.total_rewards))
        self.rects = self.ax_histo.barh(np.flip(np.arange(0, self.Nstates)) + 1 / 2,
                                         np.flip(self.probabilities[:, -1]),
                                         align="center")
        # self.rects2 = self.ax_histo3.barh(np.flip(np.arange(0, self.Nstates)) + 1 / 2,
        #                                  np.flip(self.phases[:, -1]),
        #                                  align="center")
        if self.testing is False:
            # self.ax_reward.plot([],[],marker="o",
            #                   markersize=2,markerfacecolor="blue",markeredgecolor="blue")
            self.ax_reward.plot(x, self.total_rewards)
            self.ax_histo_cumulative.plot(x, self.probs_final)
            #self.ax_histo_cumulative.plot(x, self.probs_final, color="red")

        self.ax_rew_ep = self.figure.add_subplot(gs[offset, -1])
        # self.ax_rew_ep.get_xaxis().set_visible(False)
        # self.ax_rew_ep.get_yaxis().set_visible(False)
        self.ax_rew_ep.set_xlim(0, self.tlist[-1])
        self.ax_rew_ep.set_ylim(0, 1.1)
        self.ax_rew_ep.hlines(1,0, self.tlist[-1], color="gray", linestyle="dashed")
        self.ax_rew_ep.plot(self.tlist, appo)
        self.ax_rew_ep_dual=self.ax_rew_ep.twinx()
        self.ax_rew_ep_dual.plot(self.tlist, appo,color="orange")
        self.ax_rew_ep_dual.set_xlim(0, self.tlist[-1])
        self.ax_rew_ep_dual.set_ylim(0, 1.1)

        self.ax_trace.plot(self.tlist, appo, lw=lw, color="black")
        appo_mean = np.full(int(self.T / self.t_mean), None)
        if self.num_actions==1:
            self.axes_actions.step(self.tlist, appo, lw=lw, color="red", label=r"$|\alpha|$")
            self.axes_actions.step(self.tlist_mean, appo_mean, lw=lw, color="red")
            self.axes_actions.set_ylabel(r"$|\alpha|$", labelpad=0);
        if self.num_actions > 1:
            self.axes_actions.step(self.tlist, appo, color="red", label=r"$Re[\alpha]$", alpha=0.3)
            self.axes_actions.step(self.tlist_mean, appo_mean, color="red")
            self.axes_actions.step(self.tlist, appo, color="blue", label=r"$Im[\alpha]$", alpha=0.2)
            self.axes_actions.step(self.tlist_mean, appo_mean, color="blue")
            self.axes_actions.set_ylabel(r"$Re[\alpha], Im[\alpha]$", labelpad=0);
        self.axes_actions.legend()
        self.axes_actions.set_xlim(0, self.tlist[-1])

        if self.kappa_meas>0:
            label=r"t [$1/\kappa_{meas}$]"
        else:
            label=r"t [$\alpha$]"
        if self.num_actions <= 2:
            self.axes_actions.set_xlabel(label);
        if self.num_actions > 2:
            self.axes_meas=self.figure.add_subplot(gs[offset+2, :-2])
            self.axes_meas.set_xlabel(label)
            self.axes_meas.set_xlim(0, self.tlist[-1])

            for act in range(self.num_actions-2):

                self.axes_meas.hlines(act, 0, self.tlist[-1], color="gray", linestyle="dashed", alpha=0.5)
                self.axes_meas.hlines(act + 0.5, 0, self.tlist[-1], color="gray", linestyle="dashed", alpha=0.5)
                self.axes_meas.fill_between(self.tlist,act, act + 0.5, alpha=0.1, color="gray")

                len_cycle=len(plt.rcParams['axes.prop_cycle'].by_key()['color'])
                self.axes_meas.step(self.tlist_mean, appo_mean,
                                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][act%len_cycle])
                self.axes_meas.step(self.tlist, appo, alpha=0.3,
                                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][act%len_cycle])
            self.axes_meas.set_ylim(0, self.num_actions-2.5)
            self.axes_meas.set_yticks(np.arange(0,self.num_actions-2))
            #plt.setp(self.axes_meas.get_yticklabels(), visible=False)
        # for count in range(self.N):
        #     i = int(count / 4)
        #     j = count % 4
        #     if j == 0:
        #         self.axes_obs[i, j] = self.figure.add_subplot(gs[-2 + i, j * 2:(j + 1) * 2])
        #         self.axes_obs[i, j].set_ylabel(r"Signal [$\sqrt{\kappa_{meas}}$]")
        #     else:
        #         self.axes_obs[i, j] = self.figure.add_subplot(gs[-2 + i, j * 2:(j + 1) * 2],
        #                                                        sharey=self.axes_obs[-2 + i, j - 1])
        #         plt.setp(self.axes_obs[-2 + i, j].get_yticklabels(), visible=False)
        #     if i == 0:
        #         plt.setp(self.axes_obs[-2 + i, j].get_xticklabels(), visible=False)
        #     if i == 1:
        #         self.axes_obs[i, j].set_xlabel(r"t [$1/\kappa_{meas}$]")
        #     self.axes_obs[i, j].set_xlim(0, self.tlist[-1])
        #
        #     if self.filter:
        #         self.axes_obs[i, j].plot(self.tlist, appo, lw=lw, alpha=0.3)
        #         for l in range(self.size_filters):
        #             self.axes_obs[i, j].plot(self.tlist, appo, lw=lw)
        #     else:
        #         self.axes_obs[i, j].plot(self.tlist, appo, lw=lw)
        #     if self.num_actions>2 and count<self.num_actions-2:
        #         self.axes_obs[i, j].step(self.tlist, appo, lw=3*lw, alpha=0.5, color="gray")
        #     # self.axes_integral[-2+i,j] = self.axes[-2+i,j].twinx()
        #     # self.axes_integral[-2+i,j].plot([],[] ,color="red")
        #
        #     if self.meas_type == "heterodyne":
        #         self.axes_obs[i, j].plot(self.tlist, appo, lw=lw, color="red")
        #     self.axes_obs[i, j].set_xlim(self.tlist[0], self.tlist[-1])
        #     # self.axes[-2+i,j].set_ylabel("Ch "+str(count+1), labelpad=0);

        # self.axes[self.offset,2].plot(self.tlist, appo,lw=lw, color="red", label="Purity")
        # self.axes[-2,2].legend()
        # self.axes[-1-i,j].set_xlabel(r"$t/\Delta t$", labelpad=0);

        # self.axes[-3,0].set_ylim(0,self.Nstates)
        # for i in range(4):
        #     self.axes[-3,i].set_xlim(0,self.T)
        #     self.axes[-3,i].set_xlim(0,self.T)
        #
        # for j in range(self.num_actions, 4):
        #     self.axes[-1,j].set_ylim(0,1)
        #     plt.setp(self.axes[-1,j].get_yticklabels(), visible=False)
        #     plt.setp(self.axes[-1,j].get_xticklabels(), visible=False)
        #     self.axes[-1,j].plot([0, self.T], [0, 1], 'k-', lw=lw)
        #     self.axes[-1,j].plot([0, self.T], [1, 0], 'k-', lw=lw)

        # if self.measurement_operator is None:
        #     self.axes[-2,-1].set_ylim(-1,1)
        #     plt.setp(self.axes[-2,-1].get_yticklabels(), visible=False)
        #     plt.setp(self.axes[-2,-1].get_xticklabels(), visible=False)
        #     self.axes[-2,-1].plot([0, self.T], [-1, 1], 'k-', lw=lw)
        #     self.axes[-2,-1].plot([1, self.T], [1, -1], 'k-', lw=lw)
        # if self.measurement_operator=="photocurrent":
        #     self.axes[-2,-1].set_ylim(0,1)
        # self.axes[-2,-1].step(self.tlist, appo,lw=lw)

        self.im = self.ax_trace.imshow(np.zeros((self.Nstates, self.T * self.numberPhysicsMicroSteps)),
                                       origin='lower',
                                       aspect='auto',  # get rid of this to have equal aspect
                                       vmin=0,
                                       vmax=1,
                                       alpha=0.9, interpolation="none",
                                       extent=(0, self.tlist[-1], 0, self.Nstates))

        if not self.testing:
            self.figure.suptitle('\n'.join(wrap(json.dumps(self.dic), 200)), fontsize=4)

        # plt.show(block=False)

    def render(self, mode='rgb_array', close=True):

        self.im.set_data(self.probabilities)
        # for j in range(3):
        #     self.axes[self.offset,j].lines[0].set_xdata(self.tlist)
        self.ax_trace.lines[0].set_xdata(self.tlist)
        self.ax_trace.lines[0].set_ydata(self.cavity)
        # self.axes[-3,1].lines[0].set_ydata(self.qubit)
        # self.axes[-3,2].lines[0].set_ydata(self.overlap)
        # self.axes[-3,2].lines[1].set_ydata(self.purity)

        self.ax_rew_ep.lines[0].set_xdata(self.tlist)
        self.ax_rew_ep.lines[0].set_ydata(self.fidelities)
        self.ax_rew_ep_dual.lines[0].set_xdata(self.tlist)
        self.ax_rew_ep_dual.lines[0].set_ydata(self.rewards*self.T)
        # if self.measurement_operator is not None:
        #     self.axes[-2,3].lines[0].set_xdata(self.tlist)
        #     self.axes[-2,3].lines[0].set_ydata(self.meas)

        count = 0
        self.axes_actions.lines[0].set_xdata(self.tlist)
        self.axes_actions.lines[0].set_ydata(self.actions_plot[0, :])
        self.axes_actions.lines[1].set_xdata(self.tlist_mean)
        self.axes_actions.lines[1].set_ydata(np.mean(self.actions_plot[0, :].reshape(-1, self.t_mean), axis=1))
        if self.num_actions > 1:
            self.axes_actions.lines[2].set_ydata(self.actions_plot[1, :])
            self.axes_actions.lines[3].set_xdata(self.tlist_mean)
            self.axes_actions.lines[3].set_ydata(np.mean(self.actions_plot[1, :].reshape(-1, self.t_mean), axis=1))



        self.axes_actions.set_ylim(-1.1 * np.max(np.abs(self.actions_plot[:2]))-1E-1, np.max(np.abs(self.actions_plot[:2])) * 1.1+1E-1)
        for rect, w in zip(self.rects, np.flip(self.probabilities[:, -1])):
            rect.set_width(w)
        # for rect, w in zip(self.rects2, np.flip(self.phases[:, -1])):
        #     rect.set_width(w)

        if self.testing is False:

            x = np.linspace(1, self.ep * self.ntraj, len(self.total_rewards))
            self.ax_reward.lines[0].set_data(x, self.total_rewards)
            self.ax_reward.set_xlim(1, self.ep * self.ntraj)
            self.ax_reward.collections.clear()
            self.ax_reward.fill_between(x,
                                        np.subtract(self.total_rewards, self.std_rewards),
                                        np.add(self.total_rewards, self.std_rewards), alpha=0.2, color="blue")
            self.ax_reward.set_ylim(0,
                                    max(np.add(self.total_rewards, self.std_rewards)))

            # print(np.array(self.probs_final))
            bins = 200

            self.ax_histo_cumulative.collections.clear()
            self.ax_histo_cumulative.lines[0].set_data(x, self.probs_final)
            self.ax_histo_cumulative.set_xlim(1, self.ep * self.ntraj)
            self.ax_histo_cumulative.fill_between(x,
                                        np.subtract(self.probs_final, self.std_probs_final),
                                        np.add(self.probs_final, self.std_probs_final), alpha=0.2, color="blue")
            #self.ax_histo_cumulative.lines[1].set_data(x, self.fid_finals)
        if self.num_actions > 2:
            for act in range(self.num_actions-2):
                self.axes_meas.lines[int((2*act))].set_xdata(self.tlist_mean)
                self.appo_mean = np.mean(self.actions_plot[act + 2, :].reshape(-1, self.t_mean), axis=1) / 2 + act
                self.axes_meas.lines[int((2*act))].set_ydata(self.appo_mean)
                self.axes_meas.lines[int((2 * act))+1].set_xdata(self.tlist)
                self.axes_meas.lines[int((2*act))+1].set_ydata(self.actions_plot[act + 2, :]/2+act)




        # for count in range(self.N):
        #     i = int(count / 4)
        #     j = count % 4
        #
        #     self.axes_obs[i, j].lines[0].set_xdata(self.tlist)
        #     self.axes_obs[i, j].lines[0].set_ydata(self.observations[:, 0, count])
        #     if self.filter:
        #         for l in range(self.size_filters):
        #             self.axes_obs[i, j].lines[l + 1].set_ydata(self.observations_filters[:, l, 0, count])
        #     if self.meas_type == "heterodyne":
        #         self.axes_obs[i, j].lines[1].set_ydata(self.observations[:, 1, count])
        #     maxim = np.max(np.abs(self.observations))
        #     self.axes_obs[i, j].set_ylim(-maxim * 1.1-1E-2, maxim * 1.1+1E-2)
        #
        #
        #     if self.num_actions>2 and count<self.num_actions-2:
        #         self.axes_obs[i, j].lines[-1].set_xdata(self.tlist)
        #         self.axes_obs[i, j].lines[-1].set_ydata(maxim*self.actions_plot[count+2, :])
        #     #
        #     # self.axes_integral[-2+i,j].lines[0].set_xdata(self.tlist )
        #     # self.axes_integral[-2+i,j].lines[0].set_ydata(np.array(self.integrals)[:,0,count] )
        #     # self.axes_integral[-2+i,j].set_ylim(-np.max(np.abs(self.integrals))-0.1,np.max(np.abs(self.integrals))+0.1)
        #     # print(np.mean(self.observations[:,0],axis=0))
        #
        #
        #     # self.ax_reward.set_xlim(1,len(self.total_rewards))

        if not self.testing:
            self.figure.savefig(self.direc + "/" + str(self.ep) + "_reward.png")
            if self.ep % self.save_every == 0:
                if self.folder != "":
                    if self.name_folder!="":
                        info_cleaned=self.name_folder
                    else:
                        info_cleaned="_".join(self.info.split("_")[2:])+'_'+self.direc.split("_")[-1]
                    #a_file = open(self.direc + "/../summaries/summary_" + info_cleaned + ".txt", "a")
                    #a_file.write(str(np.round(self.probs_final[-1], 5))+"\n")
                    #a_file.close()
                    # with open(self.direc + "/../summaries/summary_" + info_cleaned + ".json", "w") as fp:
                    #     print(self.__dict__)
                    #     json.dump(self.__dict__, fp, indent = 4, separators = (", ", ": "), sort_keys = True)

                    good_keys=['MsmtTrace', 'X', 'actions_plot', 'av_tries','best_reward','cavity',
                               'fidelities','fidelity','integrals','observations','observations2',
                               'probabilities','rew_init','rewards','scale','target_distrib',
                               'tlist','tlist_mean','total_rewards', 'std_rewards',
                               'probs_final', 'ep', 'ntraj', 'probs_final', 'std_probs_final', 't_mean']

                    good={x: self.__dict__[x] for x in self.__dict__ if x in good_keys}
                    np.save(self.direc + "/../summaries/summary_" + info_cleaned + ".npy", good)
                    # import pickle
                    # with open(self.direc + "/../summaries/summary_" + info_cleaned + ".json", 'w') as f:
                    #     pickle.dump(self.__dict__, f)


                    self.figure.savefig(
                        self.direc + "/../current/current_status_" + info_cleaned + ".png")
            if self.total_rewards[-1] > self.best_reward:
                self.best_reward = self.total_rewards[-1]

        # self.epoch+=1
        self.draw = False
        # self.draw=False
        # self.count=0

    # def render(self):
    #
    #     if self.t==self.T-1:
    #         self.Figure.im.set_data(self.probabilities)
    #         for j in range(3):
    #             self.Figure.axes[-2,j].lines[0].set_xdata(self.Figure.tlist)
    #         self.Figure.axes[-2,0].lines[0].set_ydata(self.cavity)
    #         self.Figure.axes[-2,1].lines[0].set_ydata(self.qubit)
    #         self.Figure.axes[-2,2].lines[0].set_ydata(self.overlap)
    #         self.Figure.axes[-2,2].lines[1].set_ydata(self.purity)
    #
    #         if self.measurement_operator is not None:
    #             self.Figure.axes[-2,3].lines[0].set_xdata(self.Figure.tlist)
    #             self.Figure.axes[-2,3].lines[0].set_ydata(self.meas)
    #
    #         maxim=np.max(np.abs(self.meas))
    #         if self.measurement_operator=="homodyne":
    #             self.Figure.axes[-2,-1].set_ylim(-maxim,maxim)
    #
    #
    #         for j in range(self.num_actions):
    #             self.Figure.axes[-1,j].lines[0].set_ydata(self.actions_plot[:,j] )
    #         self.Figure.save(self.direc + "/"+ self.info+ "_reward.png")
    #         self.reset()
