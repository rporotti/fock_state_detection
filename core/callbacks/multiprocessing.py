from stable_baselines.common.callbacks import BaseCallback
from plotting.plot import training_figure
import matplotlib.pyplot as plt
import numpy as np

class CustomCallback(BaseCallback):
    def __init__(self, mode, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.info=0
        self.direc=0
        self.mode=mode

    def _on_training_start(self) -> None:
        self.epoch = 1
        self.rewards=[]

        self.figure = plt.figure(figsize=(7,5))
        plt.rcParams.update({'font.size': 8})
        gs = self.figure.add_gridspec(4, 4)
        self.ax_main=self.figure.add_subplot(gs[:2, :])
        self.axes=np.zeros((4,4),dtype="object")
        for i in range(2):
            for j in range(4):
                self.axes[i+2,j]=self.figure.add_subplot(gs[i+2, j])
                if i==0:
                    plt.setp(self.axes[i+2,j].get_xticklabels(), visible=False)
        #ax4 = fig.add_subplot(gs[1:, -1])
        self.ax_main.set_xlabel('Epochs')
        self.ax_main.set_ylabel("Mean cumulative reward")
        self.axes[2,0].set_ylim(0,5); self.axes[2,1].set_ylim(0,)
        self.axes[2,2].set_ylim(0,); self.axes[2,3].set_ylim(-10,10)

        for i in range(4):
            self.axes[3,i].set_xlim(0,)
        self.axes[3,0].set_ylim(-1.1,0.1); self.axes[3,1].set_ylim(-1,1)
        self.axes[3,2].set_ylim(-1,1); self.axes[3,3].set_ylim(-1,1)

        self.axes[2,0].set_ylabel(r"$<a^{\dagger} a>$");
        self.axes[2,1].set_ylabel(r"$<\sigma_m^{\dagger} \sigma_m>$");
        self.axes[2,2].set_ylabel("Overlap, purity");
        self.axes[2,3].set_ylabel("Measurement");


        self.axes[3,0].set_xlabel(r"$t/\Delta t$");
        self.axes[3,1].set_xlabel(r"$t/\Delta t$");
        self.axes[3,0].set_ylabel(r"$\Delta/25g$");
        self.axes[3,1].set_ylabel(r"$\Omega_q/1E9$");
        self.axes[3,2].set_ylabel(r"$\sigma_y/1E9$");
        self.axes[3,3].set_ylabel(r"Displacement");
        self.axes[3,2].set_xlabel(r"$t/\Delta t$");
        #self.ax[1,3].set_xlabel(r"$t/\Delta t$");
        plt.tight_layout()

        plt.show(block=False)
        T=self.training_env.get_attr("T")[0]
        num_actions=self.training_env.get_attr("num_actions")[0]
        Nstates=self.training_env.get_attr("Nstates")[0]
        self.tlist = np.linspace(0,T,T)

        appo=np.zeros(T)
        self.ax_main.plot(np.arange(1,self.epoch+1),[0],lw=3)
#         ax_main.plot(np.arange(1,epoch+1),rewards,lw=3,marker="*",
#                      markersize=15,markerfacecolor="red",markeredgecolor="red")

        for i in range(4):
            if i==0: self.axes[2,i].plot(self.tlist, appo,lw=2, color="black")
            else: self.axes[2,i].plot(self.tlist, appo,lw=2)
        self.axes[2,2].plot(self.tlist, appo,lw=2, color="red", label="Purity")
        self.axes[2,2].legend()
        for i in range(num_actions):
            self.axes[3,i].step(self.tlist, appo,lw=2)

        self.axes[2,0].set_ylim(0,Nstates)
        for i in range(4):
            self.axes[2,i].set_xlim(0,T)
            self.axes[3,i].set_xlim(0,T)
        for j in range(num_actions, 4):
            self.axes[3,j].plot([0, T], [-1, 1], 'k-', lw=2)
            self.axes[3,j].plot([1, T], [1, -1], 'k-', lw=2)

        self.ax_main.tick_params(labelbottom=False)
        #####
        self.im=self.axes[2,0].imshow(np.zeros((Nstates,T)),
                            origin='bottom',
                            aspect='auto', # get rid of this to have equal aspect
                            vmin=0,
                            vmax=1,
                            alpha=0.9,
                        extent=(0,T, 0, Nstates))
        self.max_reward=-1E6
        self.epoch_max=0
            #figure.canvas.draw()

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        mode=self.mode
        if self.training_env.get_attr("t")[0]==self.training_env.get_attr("T")[0]-1:
            self.rewards.append( np.mean(np.sum(self.training_env.get_attr("rewards"),axis=1))  )
            self.ax_main.lines[0].set_xdata(np.arange(1,self.epoch+1))
            self.ax_main.lines[0].set_ydata(self.rewards)
            self.ax_main.set_ylim(min(self.rewards),max(self.rewards))
            self.ax_main.set_xlim(1,self.epoch)
            self.ax_main.set_title("Max reward: " + str(round(max(self.rewards),3)))
            self.im.set_data(self.training_env.get_attr("probabilities")[0])
            if mode=="jupyter" or mode=="script":
                if mode=="jupyter":
                    display.clear_output(wait=True)
                    display.display(self.figure)
                if mode=="script":
                    self.figure.canvas.draw_idle()
                    self.figure.canvas.start_event_loop(0.000001)
            print("Epoch: "+ str(self.epoch))
            print("Reward: "+ str(self.rewards[-1]))
            print("Folder: \"" + self.direc+"\"")
            for ax in self.axes[2]:
                ax.lines[0].set_xdata(self.tlist)
            for ax in self.axes[2]:
                ax.lines[0].set_xdata(self.tlist)
            self.axes[2,0].lines[0].set_ydata(np.mean(self.training_env.get_attr("cavity"),axis=0))
            self.axes[2,1].lines[0].set_ydata(np.mean(self.training_env.get_attr("qubit"),axis=0))
            self.axes[2,2].lines[0].set_ydata(np.mean(self.training_env.get_attr("overlap"),axis=0))
            self.axes[2,2].lines[1].set_ydata(np.mean(self.training_env.get_attr("purity"),axis=0))
            self.axes[2,3].lines[0].set_ydata(np.mean(self.training_env.get_attr("meas"),axis=0))
            print(np.shape(self.training_env.get_attr("actions")))
            for j in range(self.training_env.get_attr("num_actions")[0]):
                self.axes[3,j].lines[0].set_ydata(np.mean(self.training_env.get_attr("actions"),axis=0)[:,j] )

            self.training_env.env_method("clean")

            if self.epoch%20==0 and mode!="cluster":
                self.figure.savefig(self.direc + "/"+str(self.epoch)+ "_training.png")
            if self.epoch%10==0:
                self.figure.savefig(self.direc + "/"+ self.info+ "_reward.png")
                #_locals['self'].save(info + "/model_backup")


            if self.rewards[-1]>self.max_reward:
                self.max_reward=self.rewards[-1]
                self.epoch_max=self.epoch
                if mode!="cluster":
                    self.model.save(self.direc +"/" + self.info+"_model_backup_best")
                else:
                    self.model.save(self.direc + "/models/" +self.info+"_model_backup_best")

            # if self.epoch>self.epoch_max+200 and self.training_env.get_attr("stop")[0] is True:
            #     return False

            self.epoch+=1
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
