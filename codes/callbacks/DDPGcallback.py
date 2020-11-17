from mpi4py import MPI
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
from mpi4py import MPI
import os
import numpy as np
from plotting.plot import training_figure
from stable_baselines import logger
import matplotlib.pyplot as plt

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True
class DDPGCallback(BaseCallback):
    def __init__(self,stop_best, stop_after,check_freq,mode,args,verbose=0):
        super(DDPGCallback, self).__init__(verbose)

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
            self.x, self.y = ts2xy(load_results(self.direc_best), 'timesteps')
            if len(self.x)>0:
                window=1
                self.moving_average.append(np.mean(self.y[-window:]))
                if self.epoch%10==0 and self.epoch>0:
                    self.Figure.plot_reward(self.moving_average)
                    self.Figure.plot_data(self.training_env)
                    self.Figure.show(self.mode)

                    self.draw=False

                    print("Epoch: "+ str(self.epoch))
                    print("Folder: \"" + self.direc+"\"")
                    print("Saving figure...")

                    if self.mode!="cluster":
                        self.Figure.save(self.direc + "/"+str(self.epoch)+ "_training.png")

                        self.Figure.save(self.direc + "/"+ self.info+ "_reward.png")


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



        return True
    def _on_rollout_end(self) -> None:
        self.epoch+=1
    def _on_training_end(self) -> None:
        pass




def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
