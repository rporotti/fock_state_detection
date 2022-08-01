from environment.cavity import *
from plotting.plot import training_figure

def compare_SSE(N_states, rho_init, rho_target, ntraj, Tmax, timesteps,
             substeps, delta,omegax,wc, g, kappa, gamma, kappa_meas, measurement):


    def simulate(N_states, rho_init, rho_target, ntraj, Tmax, timesteps,
             substeps, delta,omegax, wc,g, kappa, gamma, kappa_meas, measurement):

        N_states=N_states


        measurement=measurement

        n_traj=ntraj
        T_max=Tmax
        timesteps=timesteps
        substeps=substeps

        tlist = np.linspace(0,Tmax,timesteps)
        dt=tlist[1]-tlist[0]


        kappa=kappa
        gamma=gamma
        kappa_meas=kappa_meas




        env1 = SubprocVecEnv([lambda: JCenv(debug=True,T_max=Tmax,
                                             timesteps_max=timesteps,obs=1,steps=substeps,rho_target=rho_target,
                                             rho_init=rho_init,kappa=kappa,gamma=gamma,kappa_meas=kappa_meas,
                                             measurement_operator=measurement, simulation=True,Nstates=N_states,
                                             wc=wc,g=g) for i in range(n_traj)])


        obs=env1.reset()

        reward1=[]
        state = None
        done = False

        acts=[]
        for i in range(timesteps):
            action=np.zeros((n_traj,2))
            action[:,0]=delta
            action[:,1]=omegax

            obs, r, done, _ = env1.step(action)

        return env1


    def simulate_qutip(N_states, rho_init, rho_target, ntraj, Tmax, timesteps,
                       substeps, delta,omegax,wc, g, kappa, gamma, kappa_meas,measurement):

        N_states=N_states



        measurement=measurement

        n_traj=ntraj
        T_max=Tmax
        timesteps=timesteps
        substeps=substeps



        tlist = np.linspace(0,T_max,timesteps)
        dt=tlist[1]-tlist[0]



        kappa=kappa
        gamma=gamma
        kappa_meas=kappa_meas



        a  = qt.tensor(qt.destroy(N_states), qt.qeye(2))
        sm = qt.tensor(qt.qeye(N_states), qt.destroy(2))
        sx = qt.tensor(qt.qeye(N_states), qt.sigmax())
        sy = qt.tensor(qt.qeye(N_states), qt.sigmay())
        sz = qt.tensor(qt.qeye(N_states), qt.sigmaz())
        sc_ops = [np.sqrt(kappa_meas)*a]
        #c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma)*sm]
        #c_ops=[np.sqrt(kappa) * a,np.sqrt(kappa_meas)*a]
        c_ops=[np.sqrt(kappa) * a, np.sqrt(gamma)*sm]

        rho_target=qt.tensor(rho_target,qt.qeye(2))
        e_ops = [a.dag() * a, a + a.dag(), sm.dag() * sm, rho_target]

        chi_1=0.
        chi_2=0.
        chi_3=0.
        chi_disp=0.

        H = delta * 25*g* sm.dag() * sm + g/2 * (a.dag() * sm + a * sm.dag()) + omegax*10E8/2*sx

            #chi_1*sx+chi_2*sy+chi_3*sz-1j*chi_disp*(a.dag()-a)/2




        result_ME = qt.mesolve(H, rho_init, tlist, c_ops+sc_ops, e_ops,options=qt.Options(store_states=True))

        if measurement=="homodyne":

            result_SME = qt.smesolve(H, rho_init, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                               ntraj=n_traj, nsubsteps=100, method="homodyne",
                               store_measurement=True,
                               options=qt.Options(store_states=True))
            result_SME1 = qt.smesolve(H, rho_init, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                               ntraj=1, nsubsteps=100, method="homodyne",
                               store_measurement=True,
                               options=qt.Options(store_states=True))

        if measurement=="photocurrent":
            result_SME = qt.photocurrent_mesolve(H, rho_init, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                       ntraj=ntraj, nsubsteps=1000,
                       store_measurement=True,
                       options=qt.Options(store_states=True))
            result_SME1 = qt.photocurrent_mesolve(H, rho_init, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                       ntraj=1, nsubsteps=1000,
                       store_measurement=True,
                       options=qt.Options(store_states=True),noise=0)

        return (result_ME, result_SME, result_SME1)






    def plot_figure(tlist, env1, result_ME, result_SME, result_SME1, measurement):
        tlist=result_SME.times
        dt=tlist[1]-tlist[0]

        lw=3
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)

        axes[0,0].plot(tlist, result_SME.expect[0], label=r'Qutip', lw=lw)
        axes[0,0].plot(tlist, np.mean(env1.get_attr("cavity"),axis=0), label=r'Euler', lw=lw)
        axes[0,0].plot(tlist, result_ME.expect[0], label=r'Analytical', lw=lw)




        axes[0,1].plot(tlist, result_SME.expect[2], label=r'Qutip', lw=lw)
        axes[0,1].plot(tlist, np.mean(env1.get_attr("qubit"),axis=0), label=r'Euler', lw=lw)
        axes[0,1].plot(tlist, result_ME.expect[2], label=r'Analytical', lw=lw)





        axes[0,2].plot(tlist, result_SME.expect[-1], label=r'Qutip', lw=lw)
        axes[0,2].plot(tlist, np.mean(env1.get_attr("overlap"),axis=0), label=r'Euler', lw=lw)
        axes[0,2].plot(tlist, result_ME.expect[-1], label=r'Analytical', lw=lw)


        if measurement=="homodyne":
            axes[0,3].step(tlist, np.mean(result_SME.measurement,axis=0), label=r'Qutip',lw=lw)
            axes[0,3].step(tlist, np.mean(env1.get_attr("meas"),axis=0), label=r'Euler', lw=lw)
        if measurement=="photocurrent":
            axes[0,3].step(tlist, dt*np.cumsum(np.mean(result_SME.measurement,axis=0)), label=r'Qutip',lw=lw)
            axes[0,3].step(tlist, np.cumsum(np.mean(env1.get_attr("meas"),axis=0)), label=r'Euler', lw=lw)



        axes[1,0].plot(tlist, result_SME1.expect[0], label=r'Qutip', lw=lw)
        axes[1,0].set_title("Example trajectory")
        axes[1,0].plot(tlist, env1.get_attr("cavity")[0], label=r'Euler', color="C1",lw=lw)


        axes[1,1].plot(tlist, result_SME1.expect[2], label=r'Qutip', lw=lw)
        axes[1,1].plot(tlist, env1.get_attr("qubit")[0], label=r'Euler', color="C1", lw=lw)


        axes[1,2].plot(tlist, result_SME1.expect[-1], label=r'Qutip', lw=lw)
        axes[1,2].plot(tlist, env1.get_attr("overlap")[0], label=r'Euler', color="C1", lw=lw)





        if measurement=="homodyne":
            axes[1,3].step(tlist, result_SME1.measurement[0].real, label=r'Qutip', lw=lw)
            axes[1,3].step(tlist, env1.get_attr("meas")[0], label=r'Euler', color="C1", lw=lw)
        if measurement=="photocurrent":
            axes[1,3].step(tlist, dt*np.cumsum(result_SME1.measurement), label=r'Qutip',lw=lw)
            axes[1,3].step(tlist, np.cumsum(env1.get_attr("meas")[0]), label=r'Euler', lw=lw)


        for i in range(len(axes)):
            axes[1,i].set_xlabel("t")
            for j in range(len(axes[i])):
                axes[i,j].legend()



    #     axes[0,0].plot(tlist,4*np.exp(-tlist*(kappa+kappa_meas)),color="red")
    #     axes[1,0].plot(tlist,4*np.exp(-tlist*(kappa+kappa_meas)),color="red")


    env=simulate(N_states, rho_init, rho_target, ntraj, Tmax, timesteps,
                 substeps, delta,omegax,wc, g, kappa, gamma, kappa_meas, measurement)

    (result_ME, result_SME, result_SME1)=simulate_qutip(N_states, rho_init, rho_target, ntraj, Tmax, timesteps,
                 substeps, delta,omegax,wc, g, kappa, gamma, kappa_meas, measurement)


    tlist = np.linspace(0,Tmax,timesteps)
    plot_figure(tlist, env, result_ME, result_SME, result_SME1,measurement)










def compare_RL_qutip(env1, model):

    N_states=env1.get_attr("Nstates")[0]
    rho_init=env1.get_attr("Rho_initial")[0]
    rho_target=env1.get_attr("Rho_target")[0]
    ntraj=env1.num_envs
    Tmax=env1.get_attr("T_max")[0]
    timesteps=env1.get_attr("T")[0]
    substeps=env1.get_attr("numberPhysicsMicroSteps")[0]
    wc=env1.get_attr("wc")[0]
    g=env1.get_attr("g")[0]
    kappa=env1.get_attr("kappa")[0]
    gamma=env1.get_attr("gamma")[0]
    kappa_meas=env1.get_attr("kappa_meas")[0]
    measurement=env1.get_attr("measurement_operator")[0]


    def simulate2(N_states, rho_init,rho_target, ntraj, Tmax, timesteps,
                 substeps, wc, g, kappa, gamma, kappa_meas, measurement, model):

        N_states=N_states


        measurement=measurement

        n_traj=ntraj
        T_max=Tmax
        substeps=substeps

        timesteps=timesteps


        wc = wc  # cavity frequency
        g  = g  # coupling strength

        kappa=kappa
        gamma=gamma
        kappa_meas=kappa_meas




        obs=env1.reset()

        reward1=[]
        state = None
        done = [False for _ in range(env1.num_envs)]

        acts=[]
        for i in range(timesteps):
            action, state = model.predict(obs, state=state, mask=done, deterministic=True)
            acts.append(action)
            obs, r, done, _ = env1.step(action)

        return (env1,acts)


    def simulate_qutip2(N_states, rho_init, rho_target, ntraj, Tmax, timesteps,
                       substeps, wc, g, kappa, gamma, kappa_meas,measurement, actions=None):

        N_states=N_states

        dt=Tmax/timesteps

        measurement=measurement



        wc=wc
        omega=g

        a  = qt.tensor(qt.destroy(N_states), qt.qeye(2))
        sm = qt.tensor(qt.qeye(N_states), qt.destroy(2))
        sx = qt.tensor(qt.qeye(N_states), qt.sigmax())
        sy = qt.tensor(qt.qeye(N_states), qt.sigmay())
        sz = qt.tensor(qt.qeye(N_states), qt.sigmaz())



        #rho_init=tensor(fock(N_states,0),fock(2,0))
        #rho_init=tensor(  (fock(N_states,1)+1j*fock(N_states,3))/np.sqrt(2),fock(2,0))
        rho_init=rho_init


        e_ops = [a.dag() * a, sm.dag() * sm]

        H_coupling=omega/2 * (a.dag() * sm + a * sm.dag())
        H_qubit = 25*g*sm.dag()*sm
        H_sigmax=10E8*sx/2
        H_sigmay=10E8*sy/2
        H_sigmap=sm/2
        H_sigmap=sm.dag()/2

        def H_qubit_t(t,args):
            if t>Tmax:
                t=Tmax-dt
            return actions[int(t/dt)][0]
        def H_pi_pulse_t(t,args):
            if t>Tmax:
                t=Tmax-dt
            return actions[int(t/dt)][1]

        def H_pi_pulse_t_2(t,args):
            if t>Tmax:
                t=Tmax-dt
            return actions[int(t/dt)][2]

        H_t=[H_coupling,[H_qubit,H_qubit_t],[H_sigmax,H_pi_pulse_t], [H_sigmay,H_pi_pulse_t_2]]


        sc_ops = [np.sqrt(kappa_meas)*a]
        c_ops=[np.sqrt(kappa) * a, np.sqrt(gamma)*sm]

        rho_target=qt.tensor(qt.Qobj(rho_target),qt.qeye(2))
        e_ops = [a.dag() * a, sm.dag() * sm, rho_target]

        tlist=np.linspace(0,Tmax,timesteps)
        result_ME = qt.mesolve(H_t, qt.Qobj(rho_init), tlist, c_ops+sc_ops, e_ops,options=qt.Options(store_states=True))

    #     if measurement=="homodyne":
    #         result_SME = qt.smesolve(H_t, rho_init, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
    #                            ntraj=n_traj, nsubsteps=substeps, method="homodyne",
    #                            store_measurement=True,
    #                            options=qt.Options(store_states=True))
    #     if measurement=="photocurrent":
    #         result_SME = qt.photocurrent_mesolve(H_t, rho_init, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
    #                    ntraj=ntraj, nsubsteps=substeps,
    #                    store_measurement=True,
    #                    options=qt.Options(store_states=True),method="euler")

        return result_ME




    def plot_figure2(tlist, env1, result_ME, measurement, actions):
        #tlist=result_SME.times
        dt=tlist[1]-tlist[0]

        lw=3
        fig, axes = plt.subplots(2, 4, figsize=(18*3/4, 8*3/4), sharex=True)

        #axes[0,0].plot(tlist, result_SME.expect[0], label=r'Qutip', lw=lw)
        axes[0,0].plot(tlist, env1.get_attr("cavity")[0], label=r'Euler', lw=lw)
        axes[0,0].plot(tlist, result_ME.expect[0], label=r'Analytical', lw=lw)




        #axes[0,1].plot(tlist, result_SME.expect[2], label=r'Qutip', lw=lw)
        axes[0,1].plot(tlist, env1.get_attr("qubit")[0], label=r'Euler', lw=lw)
        axes[0,1].plot(tlist, result_ME.expect[1], label=r'Analytical', lw=lw)





        #axes[0,2].plot(tlist, result_SME.expect[-1], label=r'Qutip', lw=lw)
        axes[0,0].set_ylim(0,)
        axes[0,1].set_ylim(0,)
        axes[0,2].set_ylim(0,)
        axes[0,2].plot(tlist, env1.get_attr("overlap")[0], label=r'Euler', lw=lw)
        axes[0,2].plot(tlist, result_ME.expect[2], label=r'Analytical', lw=lw)
        axes[0,0].set_ylabel(r"$<a^{\dagger}a>$")
        axes[0,1].set_ylabel(r"$<\sigma_m^{\dagger}\sigma>$")
        axes[0,2].set_ylabel("Overlap")
        axes[1,0].set_ylabel(r"$\Delta/25g$")
        axes[1,1].set_ylabel(r"$\Omega_x/1E9$")
        axes[1,2].set_ylabel(r"$\Omega_y/1E9$")
        plt.tight_layout()
    #     if measurement=="homodyne":
    #         axes[0,3].step(tlist, np.mean(result_SME.measurement,axis=0), label=r'Qutip',lw=lw)
    #         axes[0,3].step(tlist, np.mean(env1.get_attr("meas"),axis=0), label=r'Euler', lw=lw)
    #     if measurement=="photocurrent":
    #         axes[0,3].step(tlist, dt*np.cumsum(np.mean(result_SME.measurement,axis=0)), label=r'Qutip',lw=lw)
    #         axes[0,3].step(tlist, np.cumsum(np.mean(env1.get_attr("meas"),axis=0)), label=r'Euler', lw=lw)



        #axes[1,0].plot(tlist, result_SME1.expect[0], label=r'Qutip', lw=lw)
        #axes[1,0].set_title("Example trajectory")
        axes[1,0].step(tlist, actions[:,0], label=r'Euler', color="C1",lw=lw)


        #axes[1,1].plot(tlist, result_SME1.expect[2], label=r'Qutip', lw=lw)
        axes[1,1].step(tlist,  actions[:,1], label=r'Euler', color="C1", lw=lw)

        axes[1,2].step(tlist,  actions[:,2], label=r'Euler', color="C1", lw=lw)
    #     axes[1,2].plot(tlist, result_SME1.expect[-1], label=r'Qutip', lw=lw)
    #     axes[1,2].plot(tlist, env1.get_attr("overlap")[0], label=r'Euler', color="C1", lw=lw)





    #     if measurement=="homodyne":
    #         axes[1,3].step(tlist, result_SME1.measurement[0].real, label=r'Qutip', lw=lw)
    #         axes[1,3].step(tlist, env1.get_attr("meas")[0], label=r'Euler', color="C1", lw=lw)
    #     if measurement=="photocurrent":
    #         axes[1,3].step(tlist, dt*np.cumsum(result_SME1.measurement), label=r'Qutip',lw=lw)
    #         axes[1,3].step(tlist, np.cumsum(env1.get_attr("meas")[0]), label=r'Euler', lw=lw)


    #     for i in range(len(axes)):
    #         axes[1,i].set_xlabel("t")
    #         for j in range(len(axes[i])):
    #             axes[i,j].legend()



    #     axes[0,0].plot(tlist,4*np.exp(-tlist*(kappa+kappa_meas)),color="red")

    (env,acts)=simulate2(N_states, rho_init, rho_target, ntraj, Tmax, timesteps,
                 substeps, wc, g, kappa, gamma, kappa_meas, measurement, model)


    actions=np.array(acts)[:,0,:]

    result_ME=simulate_qutip2(N_states, rho_init, rho_target, ntraj, Tmax, timesteps,
                 substeps, wc, g, kappa, gamma, kappa_meas, measurement, actions)


    tlist = np.linspace(0,Tmax,timesteps)
    plot_figure2(tlist, env, result_ME,measurement,actions)










def simple_simulation(env, actions=None, model=None):

    T=env.get_attr("T")[0]
    T_max=env.get_attr("T_max")[0]
    Nstates=env.get_attr("Nstates")[0]
    numberPhysicsMicroSteps=env.get_attr("numberPhysicsMicroSteps")[0]

    obs=env.reset()

    reward=[]
    state = None
    done = [False for _ in range(env.num_envs)]

    acts=[]
    for i in range(T):

        #action=np.random.randn(n_cpu,4)
        if model is not None:
            action, state = model.predict(obs, state=state, mask=done, deterministic=True)
            obs, r, done, _ = env.step(action)
            acts.append(action[0])
        else:
            obs, r, done, _ = env.step(actions[:,i])
            acts.append(actions[0,i])
        reward.append(r)

    fig,ax=plt.subplots(2,4,figsize=(16, 8))
    plt.rcParams.update({'font.size': 14})



    #ax.plot(tlist,reward,linewidth=3,label="Reward")
    #ax[0,2].plot(tlist,env.get_attr("meas")[0],linewidth=3,label="Meas")

    tlist=np.linspace(0,T_max,T*numberPhysicsMicroSteps)
    ax[0,0].plot(tlist,env.get_attr("cavity")[0],linewidth=3,label="Cavity")
    ax[0,1].plot(tlist,env.get_attr("qubit")[0],linewidth=3,label="Cavity")
    ax[0,2].plot(tlist,env.get_attr("overlap")[0],linewidth=3,label="Cavity")
    ax[0,2].plot(tlist,env.get_attr("purity")[0],linewidth=3,label="Cavity", color="red")
    ax[0,3].plot(tlist,env.get_attr("meas")[0],linewidth=3)
    ax[0,0].imshow(env.get_attr("probabilities")[0],
                        origin='bottom',
                        aspect='auto', # get rid of this to have equal aspect
                        vmin=0,
                        vmax=1,
                        alpha=0.9,
                    extent=(0,T_max, 0, Nstates))

    ax[0,1].set_ylim(0,1)
    ax[0,2].set_ylim(0,1)
    titles=[r"$\omega_q$",r"$\sigma_x$",r"$\sigma_y$", "Displacement"]

    tlist2=np.linspace(0,T_max,T)
    for i in range(env.get_attr("num_actions")[0]):
        ax[1,i].set_title(titles[i])
        ax[1,i].step(tlist2,np.array(acts)[:,i])
    return env, env.get_attr("actions_plot")[0]
    #ax.set_xlabel("timesteps")
    #ax.set_ylabel("Mean reward")
    #ax.legend()



def compare_with_qutip_2(args,rho_init, rho_target, model=None):
    args=vars(args)
    env = CavityEnv(args,rho_init=rho_init, rho_target=rho_target)
    Figure=training_figure(env=env, with_reward=False, args=args)


    done = False

    acts=[]
    dict1={}
    N_envs=51
    cavity=np.zeros((N_envs,env.T*env.numberPhysicsMicroSteps))
    qubit=np.zeros((N_envs,env.T*env.numberPhysicsMicroSteps))
    overlap=np.zeros((N_envs,env.T*env.numberPhysicsMicroSteps))
    meas=np.zeros((N_envs,env.T*env.numberPhysicsMicroSteps))

    for n in range(N_envs):
        obs=env.reset()
        for i in range(env.T):
            if model is None:
                action= np.zeros(env.num_actions)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            cavity[n,:]=env.cavity
            qubit[n,:]=env.qubit
            overlap[n,:]=env.overlap
            meas[n,:]=env.meas

    cavity=np.mean(cavity,axis=0)
    qubit=np.mean(qubit,axis=0)
    overlap=np.mean(overlap,axis=0)
    meas=np.mean(meas,axis=0)*1E4

    dict1={"cavity":cavity, "qubit":qubit,"overlap":overlap, "meas":meas}


    tlist = np.linspace(0,args["T_max"],args["timesteps"])
    dt=tlist[1]-tlist[0]
    sc_ops = [np.sqrt(env.kappa_meas)*env.a]
    c_ops=[np.sqrt(env.kappa) * env.a, np.sqrt(env.gamma)*env.sm]

    rho_goal=qt.tensor(rho_target,qt.qeye(2))
    e_ops = [env.a.dag() * env.a, env.sm.dag() * env.sm,rho_goal]

    H = env.g/2* (env.a.dag()*env.sm+env.a*env.sm.dag())
    #result_ME = qt.mesolve(H, rho_init, tlist, c_ops+sc_ops, e_ops,options=qt.Options(store_states=True))

    result_SME1 = qt.smesolve(H, rho_init, tlist, c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                       ntraj=N_envs, nsubsteps=100, method="homodyne",
                       store_measurement=True,
                       options=qt.Options(store_states=True))


    dict2={"cavity":result_SME1.expect[0], "qubit":result_SME1.expect[1],
    "overlap":result_SME1.expect[2],"meas":np.mean(result_SME1.measurement,axis=0)}

    Figure.plot_lines(dict1,dict2)
    Figure.save("noname.png")
