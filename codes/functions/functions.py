import datetime
import pytz
import os
import sys
import subprocess
import qutip as qt
from mpi4py import MPI

def create_dir(args):
    
    info=create_info(args)

    hour = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%H.%M.%S")
    date = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y-%m-%d")

    rank = MPI.COMM_WORLD.Get_rank()
    mode=args["mode"]
    main_folder=args["main_folder"]
    folder=date+"_"+args["folder"]


    if args["library"]=="SB":
        if mode!="cluster":
            if mode=="jupyter":
                direc=main_folder+"jupyter_"+info
            if mode=="script":
                if folder!="":
                    direc=main_folder+folder+"/"+info
                else:
                    direc=main_folder+info
                if os.path.isdir(direc):
                    _add="_"+hour
#                    i=2
#                    while os.path.isdir(direc+"_v"+str(i)):
#                        i+=1
#                    info+="_v"+str(i)
        else:
            direc="../simulations_cluster/"


    if rank==0:
        if os.path.isdir(direc) is True:
            direc+=_add

        


    return direc



def create_info(args):
    rank = MPI.COMM_WORLD.Get_rank()
    mode=args["mode"]
    folder=args["folder"]
    hour=datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%H.%M.%S")
    now=datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%d-%m-%Y_%H.%M.%S")
    date=datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y-%m-%d")

    info=date+"_cavity"
    appo=""
    exclude_list=["HER","animation","load_parameters","same",
     "mode","mpi", "lstm","stop_best","fixed_seed", "folder","filter","discrete","capped_to_zero"]
    for i in range(1,len(sys.argv)):
        if sys.argv[i].startswith("--"):
            line=sys.argv[i].split("--")[1]
            if line not in exclude_list:
                appo+="_"+line+sys.argv[i+1]
            if line=="lstm":
                appo+="_lstm"
    if appo=="":
        appo="_standard"
    info+=appo

    if args["HER"]:
        info+="_HER"

    return info





def print_info(args, direc, rank=0):
    if rank==0:
        file = open(direc+"/info.txt","w")
        for item, value in args.items():  # dct.iteritems() in Python 2
            file.write("{}: {}\n".format(item, value))

        file.write("\n\n")
        file.write(sys.argv[0] +" ")
        for i in range(1,len(sys.argv)):
            file.write(sys.argv[i]  +" ")
        version = subprocess.check_output(["git", "describe"]).strip().decode('utf-8')
        file.write("\ngit version: "+version)


    #print(sys.argv)



def split_string(N_states,s):
    s=str(s)
    if s.isdigit():
        return qt.basis(N_states,int(s))
    else:
        l=s.split("+")
        state=0
        for item in l:
            state+=qt.basis(N_states,int(item))
        return state.unit()
