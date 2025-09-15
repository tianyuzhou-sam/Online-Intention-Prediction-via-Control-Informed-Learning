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
dynsys = Env.Dog()
wx = 100
wv = 1
wd = 0.07
wu1 = 1
wu2 = 3
wu3 = 1

goal = [2,-1.0]
goal_v = [0,0]

dynsys.initDyn()
dynsys.initCost(wx=wx, wv=wv, wd=wd, wu1=wu1, wu2=wu2, wu3=wu3, goal=goal)
dt = 0.1
horizon = 100
init_state = [-2,-1.0,0]
true_parameter = [wx, wv, wd, wu1, wu2, wu3, goal]
dir = 'example/'
demoFile = 'dog_demos.mat'

generateTraj.generateTraj(dynsys, init_state, dt, horizon, true_parameter, dir, demoFile, saveFlag)

