import qutip as qt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

def init_figure(env,xwigner):
    global fig, axes
    plt.rcParams.update({'figure.dpi': 300})
    #plt.rcParams.update({'font.size': 4})
    scale=1
    fig=plt.figure(figsize=(6*scale, 3*scale), constrained_layout=True)


    grid= plt.GridSpec(4, 8, figure=fig)
    axes=np.zeros(6,dtype=object)
    axes[1]=fig.add_subplot(grid[:2, :2])
    axes[2]=fig.add_subplot(grid[2:4, :2], sharey=axes[1])
    axes[3]=fig.add_subplot(grid[:2, 2:])
    for i in range(2):
        axes[i+1].plot([], [],lw=1)
    labels=["Detuning", r"$\sigma_x$", r"$\sigma_y$"]
    for i in range(env.num_actions):
        axes[3].step([], [],lw=1, label=labels[i])
    for i in range(3):
        axes[i+1].set_xlim(0,env.T*env.numberPhysicsMicroSteps)

    appo = np.empty((env.T*env.numberPhysicsMicroSteps,5))
    appo[:] = np.NaN

    W = qt.wigner(qt.Qobj(env.RhoGoal), xwigner, xwigner)
    wmap = qt.wigner_cmap(W)  # Generate Wigner colormap
    nrm = matplotlib.colors.Normalize(-W.max(), W.max())
    axes[1].set_ylim(0,1)
    axes[2].set_ylim(0,1)
    axes[3].set_ylim(-1.1,1.1)
    axes[4]=fig.add_subplot(grid[2:4, 2:5], adjustable='box', aspect=1.)
    axes[5]=fig.add_subplot(grid[2:4, 5:8], adjustable='box', aspect=1.)
    plt1 = axes[4].pcolor(xwigner, xwigner, W, cmap=matplotlib.cm.RdBu, norm=nrm)
    for i in range(5):
        if i+1!=3:
            axes[i+1].set_xticklabels([])
            axes[i+1].set_xticks([]); axes[i+1].set_xticks([], minor=True)
            axes[i+1].set_yticklabels([])
            axes[i+1].set_yticks([]); axes[i+1].set_yticks([], minor=True)
    axes[2].set_title("Fidelity");
    axes[3].set_title("Actions");
    axes[3].set_ylabel(r"$\Delta/25g$, $\sigma_x/g$");
    axes[1].set_title("Qubit excitation");
    axes[1].set_ylabel(r"$\langle \sigma_m^{\dagger} \sigma_m\rangle$");
    axes[4].set_title("Target state");
    axes[5].set_title("RL");
    #plt.tight_layout()
    axes[4].set_xticklabels([])
    axes[4].set_yticklabels([])
    axes[4].set_xticks([]); axes[4].set_xticks([], minor=True)
    axes[4].set_yticks([]); axes[4].set_yticks([], minor=True)
    axes[3].legend(loc=1)
    axes[1].set_yticklabels([0,0.5,1])
    axes[1].set_yticks([0,0.5,1])
    return fig


def create_wigners(env, xwigner):
    wigners=np.zeros((len(env.Rho_int),len(xwigner),len(xwigner)))
    for i in range(len(env.Rho_int)):
        wigners[i]=qt.wigner(qt.Qobj(env.Rho_int[i]),xwigner,xwigner)
    return np.array(wigners)


def animate(i, env, wigners, xwigner):
    global axes
    if i%(len(wigners)/10)==0 and i>0:
        print(str((i/(len(wigners)/10))*10)+"%")
    #axes[4].clear()

    for j in range(2):
        axes[j+1].lines[0].set_xdata(np.arange(0,i,1))
    axes[1].lines[0].set_ydata(np.array(env.qubit)[:i])
    axes[2].lines[0].set_ydata(np.array(env.overlap)[:i])
    scale=[1,1,1]
    for j in range(env.num_actions):
        axes[3].lines[j].set_xdata(np.arange(0,i,1))
        axes[3].lines[j].set_ydata(np.array(env.actions_plot)[:i,j]*scale[j])


    nrm = matplotlib.colors.Normalize(-wigners[i,:,:].max(), wigners[i,:,:].max())

    axes[5].pcolormesh(xwigner, xwigner, wigners[i], cmap=matplotlib.cm.RdBu, norm=nrm)
    #axes[5].set_title("RL");
#         plt1 = axes2.contourf(xwigner,xwigner,np.array(wigners)[i,:,:], 100,cmap=cm.RdBu,
#                               vmin=-np.max(np.abs(wigners)),vmax=np.max(np.abs(wigners)))




def plot_animation(env, filename=None):
    global fig
    N=100
    xwigner=np.linspace(-2.5,2.5,N)



    wigners=create_wigners(env, xwigner)
    fig=init_figure(env,xwigner)
    print("Start producing animation...")
    interval=20/len(wigners)*10000 #20 seconds
    frames=len(wigners)

    anim = animation.FuncAnimation(fig,animate,
                    frames=frames,interval=interval,blit=False,
                    fargs=(env,wigners,xwigner))

    #plt.close(anim._fig)
    #HTML(anim.to_html5_video())

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=len(wigners)/5, metadata=dict(artist='Me'), bitrate=1800)
    #plt.show()
    if filename is not None:
        anim.save(filename+".mp4", writer=writer)
    else:
        anim.save("noname.mp4", writer=writer)

















def video_wigners(env, filename=None):
    import matplotlib as mpl
    from matplotlib import cm
    len_wigners=len(wigners)

    def animate(i):
        if i%(len_wigners/10)==0 and i>0:
            print(str((i/(len_wigners/10))*10)+"%")
        axes2.clear()
        nrm = mpl.colors.Normalize(-np.array(wigners)[i,:,:].max(), np.array(wigners)[i,:,:].max())

        axes2.pcolormesh(xwigner, xwigner, wigners[i], cmap=cm.RdBu, norm=nrm)

#         plt1 = axes2.contourf(xwigner,xwigner,np.array(wigners)[i,:,:], 100,cmap=cm.RdBu,
#                               vmin=-np.max(np.abs(wigners)),vmax=np.max(np.abs(wigners)))

    def plot_wigner(interval):
        plt.xlabel(r'x')
        plt.ylabel(r'y')
        anim = animation.FuncAnimation(fig2,animate,frames=len_wigners,interval=interval,
            blit=False, repeat=True)
        return anim

    interval=5000/len_wigners
    fig2, axes2 = plt.subplots(1, 1, figsize=(10, 10))
    anim=plot_wigner(interval)
    plt.close(anim._fig)
    #HTML(anim.to_html5_video())

    Writer = animation.writers['pillow']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    if filename is not None:
        anim.save(filename, writer=writer)
    return anim
