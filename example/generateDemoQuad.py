import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
import ImitationLearning
import Env
import generateTraj
inf = 1e20

# ------------------------------ Set up dynamic system ------------------------------
saveFlag = False
dynsys = Env.Quadrotor()
# Jx = 6.85*10**(-5)
# Jy = 9.2*10**(-5)
# Jz = 13.66*10**(-4)
# mass = 0.068
# l = 0.062
Jx = 1
Jy = 1
Jz = 1
mass = 1
l = 1
wr=10
wv=10
wq=100
ww=10
wd=0.0002
wthrust=0.1

goal = [2,-.0,.6,0,0,0,0,0,0,1,0,0,0]

dynsys.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=l)
dynsys.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wd=wd, goal=[goal[0],goal[1],goal[2]], wthrust=wthrust)
dt = 0.1
horizon = 100
init_state = [-2,-.0,0.6,0.1,0.2,0,0,0,0,1,0,0,0]
true_parameter = [wr, wv, wq, ww, wd, [goal[0],goal[1],goal[2]]]
dir = 'example/'
demoFile = 'quad_demos.mat'

generateTraj.generateTraj(dynsys, init_state, dt, horizon, true_parameter, dir, demoFile, saveFlag)

