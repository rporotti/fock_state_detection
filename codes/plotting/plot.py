import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import matplotlib
from textwrap import wrap
import json
matplotlib.use('Agg')



class training_figure():
    def __init__(self, env, args,with_reward=True):

        dpi=300
        lw=1
        #plt.rcParams.update({'font.size': 3})
        plt.rcParams.update({'figure.dpi': dpi})

        if with_reward:
            self.figure = plt.figure(figsize=(8,6),constrained_layout=True)
            self.gs = self.figure.add_gridspec(4, 4)

            self.ax_main=self.figure.add_subplot(self.gs[:2, :])
            self.ax_main.set_xlabel('Epochs')
            self.ax_main.set_ylabel("Mean cumulative reward")
            #self.ax_main.tick_params(axis="x", pad=0, length=2)

        else:
            self.figure = plt.figure(figsize=(8,3),constrained_layout=True)
            self.gs = self.figure.add_gridspec(2, 4)
            self.ax_main=0

        self.axes=np.zeros((4,4),dtype="object")
        for i in range(2):
            for j in range(4):
                self.axes[-2+i,j]=self.figure.add_subplot(self.gs[-2+i, j])
                if i==0:
                    plt.setp(self.axes[-2+i,j].get_xticklabels(), visible=False)
        #ax4 = fig.add_subplot(gs[1:, -1])

        self.axes[-2,0].set_ylim(0,5); self.axes[-2,1].set_ylim(0,1.1)
        self.axes[-2,2].set_ylim(0,1.1);

        for i in range(env.num_actions):
            self.axes[-1,i].set_xlim(0,)
            self.axes[-1,i].set_ylim(env.lim_actions[i][0]-0.1,env.lim_actions[i][1]+0.1);


        self.axes[-2,0].set_ylabel(r"$<a^{\dagger} a>$", labelpad=0);
        self.axes[-2,1].set_ylabel(r"$<\sigma_m^{\dagger} \sigma_m>$", labelpad=0);
        self.axes[-2,2].set_ylabel("Overlap, purity", labelpad=0);
        self.axes[-2,3].set_ylabel("Measurement");


        self.axes[-1,0].set_xlabel(r"$t/\Delta t$"); self.axes[-1,0].set_ylabel(r"$\Delta/25g$", labelpad=0);
        self.axes[-1,1].set_xlabel(r"$t/\Delta t$"); self.axes[-1,1].set_ylabel(r"$\sigma_x/1E9$", labelpad=0);
        self.axes[-1,2].set_xlabel(r"$t/\Delta t$"); self.axes[-1,2].set_ylabel(r"$\sigma_y/1E9$", labelpad=0);
        self.axes[-1,3].set_ylabel(r"Displacement");

        #self.ax[1,3].set_xlabel(r"$t/\Delta t$");

        self.tlist = np.linspace(0,env.T,env.T*env.numberPhysicsMicroSteps)

        appo=np.full(env.T*env.numberPhysicsMicroSteps,None)
        if with_reward:
            self.ax_main.plot(np.arange(1,2),[0],lw=lw)
            self.ax_main.plot(np.arange(1,2),[0],lw=lw, color="red")

    #         ax_main.plot(np.arange(1,epoch+1),rewards,lw=3,marker="*",
    #                      markersize=15,markerfacecolor="red",markeredgecolor="red")

        for i in range(3):
            if i==0: self.axes[-2,i].plot(self.tlist, appo,lw=lw, color="black")
            else: self.axes[-2,i].plot(self.tlist, appo,lw=lw)
        self.axes[-2,2].plot(self.tlist, appo,lw=lw, color="red", label="Purity")
        #self.axes[-2,2].legend()
        for i in range(env.num_actions):
            self.axes[-1,i].step(self.tlist, appo,lw=lw)

        self.axes[-2,0].set_ylim(0,env.Nstates)
        for i in range(4):
            self.axes[-2,i].set_xlim(0,env.T)
            self.axes[-1,i].set_xlim(0,env.T)
        for j in range(env.num_actions, 4):
            self.axes[-1,j].set_ylim(0,1)
            plt.setp(self.axes[-1,j].get_yticklabels(), visible=False)
            plt.setp(self.axes[-1,j].get_xticklabels(), visible=False)
            self.axes[-1,j].plot([0, env.T], [0, 1], 'k-', lw=lw)
            self.axes[-1,j].plot([0, env.T], [1, 0], 'k-', lw=lw)

        if env.measurement_operator is None:
            self.axes[-2,-1].set_ylim(-1,1)
            plt.setp(self.axes[-2,-1].get_yticklabels(), visible=False)
            plt.setp(self.axes[-2,-1].get_xticklabels(), visible=False)
            self.axes[-2,-1].plot([0, env.T], [-1, 1], 'k-', lw=lw)
            self.axes[-2,-1].plot([1, env.T], [1, -1], 'k-', lw=lw)
        if env.measurement_operator=="photocurrent":
            self.axes[-2,-1].set_ylim(0,1)
        self.axes[-2,-1].step(self.tlist, appo,lw=lw)


        self.im=self.axes[-2,0].imshow(np.zeros((env.Nstates,env.T)),
                            origin='bottom',
                            aspect='auto', # get rid of this to have equal aspect
                            vmin=0,
                            vmax=1,
                            alpha=0.9,
                        extent=(0,env.T, 0, env.Nstates))

        if isinstance(args,dict) is False:
            dic=vars(args)
        else: dic=args
        self.figure.suptitle('\n'.join(wrap(json.dumps(dic), 200)), fontsize=4)
        #self.gs.tight_layout(self.figure,rect=[0, 0.0, 1.1, 1.])
        #self.ax_main.text(0.5, 1, vars(args), ha='right',va="top", wrap=True,
        #fontsize=20, transform=self.ax_main.transAxes)
        #self.figure.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.show(block=False)

    def plot_data(self, training_env):
        self.im.set_data(training_env.probabilities)
        for j in range(3):
            self.axes[-2,j].lines[0].set_xdata(self.tlist)
        self.axes[-2,0].lines[0].set_ydata(training_env.cavity)
        self.axes[-2,1].lines[0].set_ydata(training_env.qubit)
        self.axes[-2,2].lines[0].set_ydata(training_env.overlap)
        self.axes[-2,2].lines[1].set_ydata(training_env.purity)

        if training_env.measurement_operator is not None:
            self.axes[-2,3].lines[0].set_xdata(self.tlist)
            self.axes[-2,3].lines[0].set_ydata(training_env.meas)

        maxim=np.max(np.abs(training_env.meas))
        if training_env.measurement_operator=="homodyne":
            self.axes[-2,-1].set_ylim(-maxim,maxim)


        for j in range(training_env.num_actions):
            self.axes[-1,j].lines[0].set_ydata(training_env.actions_plot[:,j] )
    def plot_reward(self, reward):
        self.ax_main.lines[1].set_xdata(np.arange(1,len(reward)+1))
        self.ax_main.lines[1].set_ydata(reward)
        self.ax_main.set_ylim(min(reward),max(reward))
        self.ax_main.set_xlim(1,len(reward))
        #self.ax_main.set_title("Max reward: " + str(round(max(reward),3)))

    def plot_lines(self, *args):
        for item in args:
            self.axes[-2,0].plot(self.tlist, item["cavity"])
            self.axes[-2,1].plot(self.tlist, item["qubit"])
            self.axes[-2,2].plot(self.tlist, item["overlap"])


            self.axes[-2,3].plot(item["meas"])
            maxim=np.max(np.abs(item["meas"]))
            self.axes[-2,-1].set_ylim(-maxim,maxim)
        #
        #
        # for j in range(training_env.num_actions):
        #     self.axes[-1,j].lines[0].set_ydata(training_env.actions_plot[:,j] )
    def plot_qutip(self, result, overlap):
        #self.im.set_data(training_env.probabilities)

        self.axes[-2,0].plot(result.expect[0], label="Qutip")


        self.axes[-2,1].plot(result.expect[1], label="Qutip")
        self.axes[-2,2].plot(overlap, label="Qutip")

        #self.axes[-2,2].lines[1].set_ydata(training_env.purity)
        #
        # if training_env.measurement_operator is not None:
        #     self.axes[-2,3].lines[0].set_xdata(self.tlist)
        #     self.axes[-2,3].lines[0].set_ydata(training_env.meas)
        #
        # maxim=np.max(np.abs(training_env.meas))
        # if training_env.measurement_operator=="homodyne":
        #     self.axes[-2,-1].set_ylim(-maxim,maxim)
        #
        #
        # for j in range(training_env.num_actions):
        #     self.axes[-1,j].lines[0].set_ydata(training_env.actions_plot[:,j] )


    def save(self,path):
        self.figure.savefig(path,dpi=300, bbox_inches='tight')
    def show(self,mode):

        if mode=="jupyter" or mode=="script":
            if mode=="jupyter":
                display.clear_output(wait=True)
                display.display(self.figure)
            # if mode=="script":
            #     self.figure.canvas.draw_idle()
            #     self.figure.canvas.start_event_loop(0.000001)
                #plt.pause(0.000001)



def plot_env(env, args, model=None):
    Figure=training_figure(env=env,with_reward=False, args=args)

    obs=env.reset()
    done = False

    acts=[]
    #env.render(close=False)
    for i in range(env.T):
        if model is None:
            action=np.zeros(env.num_actions)
        else:
            action, state = model.predict(obs, deterministic=True)
        obs, r, done, _ = env.step(action)
    #env.render()
    #plt.show()
    Figure.plot_data(env)
    Figure.save("noname.png")
    #plt.savefig("noname.png")
    #plt.show(block=False)

    return env
