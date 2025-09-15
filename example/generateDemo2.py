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
wpath = 1
wv = 1
wd = 1
wu1 = 1
wu2 = 3
wu3 = 1
# wgoal = 100
# w_xsq = 0
# w_x = -1
# w_ysq = -0
# w_y = -0
# wxy = -1.0
# wu = 1

goal = [2,0.0]

dynsys.initDyn2()
dynsys.initCost(wx=wx, wv=wv, wd=wd, wu1=wu1, wu2=wu2, wu3=wu3, goal=goal)
# dynsys.initCostPoly(wgoal=wgoal, w_xsq=w_xsq, w_x=w_x, w_ysq=w_ysq, w_y=w_y, wxy=wxy, wu=wu, goal=goal)
dt = 0.1
horizon = 100
init_state = [-2,-0.0,0,0.0,0.0]
true_parameter = [wx, wv, wd, wu1, wu2, wu3, goal[0], goal[1]]
dir = 'example/'
demoFile = 'dog_demos.mat'

generateTraj.generateTraj(dynsys, init_state, dt, horizon, true_parameter, dir, demoFile, saveFlag)

