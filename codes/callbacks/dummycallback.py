from mpi4py import MPI
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
from mpi4py import MPI
import os
import numpy as np
from plotting.plot import training_figure
from stable_baselines import logger





class DummyCallback(BaseCallback):
    def __init__(self,stop_best, stop_after,check_freq,mode,args,verbose=0):
        super(DummyCallback, self).__init__(verbose)

        self.info=0
        self.direc=0
        self.mode=mode
        self.check_freq=check_freq
        self.moving_average=[]
        self.args=args
        self.stop_best=stop_best
        self.stop_after=stop_after

        self.direc_best = 0
        self.best_mean_reward = -np.inf
        self.best_epoch=0
    def _on_training_start(self) -> None:

        self.epoch = 0
        self.all_rewards=[]
        self.rewards=[]
        self.draw=False
        rank = MPI.COMM_WORLD.Get_rank()
        if rank==0:
            self.Figure=training_figure(with_reward=True,env=self.training_env, args=self.args)

            self.max_reward=-1E6
            self.epoch_max=0
                    #figure.canvas.draw()
        pass
    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        rank = MPI.COMM_WORLD.Get_rank()
        if rank==0 and self.training_env.t==self.training_env.T:
            if len(self.moving_average)>0 and self.moving_average[-1] > self.best_mean_reward :
                self.best_epoch=self.epoch
                self.best_mean_reward = self.moving_average[-1]
                self.Figure.plot_reward(self.moving_average)
                self.Figure.plot_data(self.training_env)
                self.Figure.show(self.mode)

                if self.mode!="cluster":
                    if self.verbose > 0:
                      print("Saving new best model at {} timesteps".format(self.x[-1]))
                      print("Saving new best model to {}.zip".format(os.path.join(self.direc_best, 'best_model')))
                      self.model.save(os.path.join(self.direc_best, 'best_model'))
                      self.Figure.save(self.direc + "/best_reward.png")
                else:
                    self.Figure.save(self.direc + "/" + self.info+ ".png")


        if rank==0 and self.draw and self.training_env.t==self.training_env.T:
            if self.epoch%50==0:
                self.Figure.plot_reward(self.moving_average)
                self.Figure.plot_data(self.training_env)
                self.Figure.show(self.mode)

                self.draw=False

                print("Epoch: "+ str(self.epoch))
                print("Folder: \"" + self.direc+"\"")
                print("Saving figure...")

                self.training_env.clean()

                if self.mode!="cluster":
                    self.Figure.save(self.direc + "/"+str(self.epoch)+ "_training.png")

                    self.Figure.save(self.direc + "/"+ self.info+ "_reward.png")
            #self.epoch+=1
        if self.stop_after>-1:
            if self.epoch>self.stop_after:
                return False
        if self.stop_best:
            if self.epoch>self.best_epoch+3:
                return False

        return True

    def _on_rollout_end(self) -> None:
        self.epoch+=1
        rank = MPI.COMM_WORLD.Get_rank()
        if rank==0:
            self.x, self.y = ts2xy(load_results(self.direc_best), 'timesteps')
            if len(self.x)>0:
                window=10
                self.moving_average.append(np.mean(self.y[-window:]))


        if self.epoch%10==0:
            self.draw=True

    def _on_training_end(self) -> None:
        pass
