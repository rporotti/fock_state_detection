import sys
from stable_baselines import PPO1
from codes.plotting.plot import plot_env
from codes.plotting.animation import plot_animation
from codes.environment.multichannel import SimpleCavityEnv

def load_info(direc):
    with open(direc+"/info.txt") as infile:
        args={}
        line=infile.readline()
        while(line!="\n"):
            data = line.split(": ")

            name, value = data[0], data[1].split("\n")[0]
            if value.isdigit():
                value=int(value)
            else:
                try:
                    value=float(value)
                except (ValueError, TypeError):
                    if value=="None": value=None
                    if value=="True": value=True
                    if value=="False": value=False

            args.update( {name : value} )
            line=infile.readline()
    print("Values loaded from "+ direc+":")
    print(args)
    return args

def check_policy(rho_init, rho_target,folder_policy):
    args=load_info(folder_policy)
    animation=False

    appo_dic={}
    exclude_list=["animation","load_parameters","same"]
    for i in range(2,len(sys.argv)):
        if sys.argv[i].startswith("--"):
            name=sys.argv[i].split("--")[1]
            if name=="animation": animation=True
            if name not in exclude_list:
                value=sys.argv[i+1]
                if value.isdigit():
                    value=int(value)
                else:
                    try:
                        value=float(value)
                    except (ValueError, TypeError):
                        value=str(value)
                appo_dic.update({name: value })
    if bool(appo_dic):
        print("CHANGED THE FOLLOWING VALUES:")
        for key, value in appo_dic.items():
            print(key+": "+str(value))
            args.update({key: value})


    env = CavityEnv(args, rho_init=rho_init, rho_target=rho_target)

    model=PPO1.load("../simulations/"+folder_policy+"/progress/best_model.zip",env)
    env=plot_env(env, args, model)
    if animation is True:
        plot_animation(env)
