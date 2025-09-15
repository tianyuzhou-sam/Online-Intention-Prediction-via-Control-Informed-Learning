import numpy as np
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt 
import os, shutil
import sys
import time
sys.path.append(os.getcwd() + '/src/')
import Env
import OC

class generateTraj:
    def __init__(self, dynsys=None, init_state=None, dt=0, horizon=0, true_parameter=None, dir="", demoFile="", saveFlag=False):
        # ------------------------------ initialize Classes ------------------------------
        sysoc = OC.OCSys()
        sysoc.setControlVariable(dynsys.U)
        sysoc.setStateVariable(dynsys.X)
        dyn = dynsys.X + dt * dynsys.f
        sysoc.setDyn(dyn)
        sysoc.setPathCost(dynsys.path_cost)
        sysoc.setFinalCost(dynsys.final_cost)
        
        traj = sysoc.ocSolver(ini_state=init_state, horizon=horizon)

        state_traj = traj['state_traj_opt']
        control_traj = traj['control_traj_opt']

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(len(state_traj[0]),1)
        for idx in range(len(state_traj[0])):
            axs[idx].plot(iter, state_traj[:,idx])
            axs[idx].set_ylabel("x"+str(idx+1))
        axs[-1].set_xlabel("Iteration")
        axs[0].set_title("State Trajectory")

        iter = [*range(len(control_traj))]
        if len(control_traj[0]) == 1:
            fig, axs = plt.subplots()
            axs.plot(iter, control_traj)
            axs.set_ylabel("u")
            axs.set_xlabel("Iteration")
            axs.set_title("Control Trajectory")
        else:
            fig, axs = plt.subplots(len(control_traj[0]),1)
            for idx in range(len(control_traj[0])):
                axs[idx].plot(iter, control_traj[:,idx])
                axs[idx].set_ylabel("x"+str(idx+1))
            axs[-1].set_xlabel("Iteration")
            axs[0].set_title("Control Trajectory")

        fig, axs = plt.subplots(1,1)
        axs.plot(state_traj[:,0], state_traj[:,1])
        # axs.plot(true_parameter[7], true_parameter[8], 'r*')
        # axs.set_xlim(-3,3)
        # axs.set_ylim(-3,3)
        
        plt.show()



        if saveFlag:
            sio.savemat(dir+demoFile, {'trajectories': traj, 'dt': dt, 'true_parameter': true_parameter})

            


    