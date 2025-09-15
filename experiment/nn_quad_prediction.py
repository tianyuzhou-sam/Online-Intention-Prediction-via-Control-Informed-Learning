import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
import transforms3d
sys.path.append(os.getcwd() + '/experiment')
import ImitationLearning
import Env

init_state = [-2,-0,.6,0.1,0.2,0,0,0,0,1,0,0,0]
# init_state = [-2,-.0,0.6,0.1,0.2,0,0,0,0,1,0,0,0]
Jx = 1
Jy = 1
Jz = 1
mass = 1
l = 1
wr=10
wv=10
wq=100
ww=10
wd=0.0004

dt = 0.1
horizon = 100

pred_init = [-2,-0.0,0.6]
# pred_init = [-2,-.0,0.6]

goal = [2,0.,.6]

true_theta = [Jx, Jy, Jz, mass, l, wr, wv, wq, ww, wd]


# ------------------------------ Set up dynamic system ------------------------------
project = 'quad'
saveFlag = True
dynsys = Env.Quadrotor()
dynsys.initDyn()
dynsys.initNeuralDyn(hidden_layers=[128])
dynsys.initCost()

theta = np.random.random(dynsys.n_auxvar)-0.5
            
system = ImitationLearning.ImitationLearningNN(project, init_state, true_theta, dynsys, dt, horizon, pred_init, saveFlag)
system.set_iteration(1)
system.initialize_nn_parameter(theta)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(13) * 0.000000000001
multiplier = 1000
P[-3,-3] = P[-3,-3]*multiplier
P[-2,-2] = P[-2,-2]*multiplier
P[-1,-1] = P[-1,-1]*multiplier
Q = np.eye(13) * 0.
R = np.eye(2) * 0.0000000001

# P = np.eye(13) * 0.000000000001
# multiplier = 1000
# P[-3,-3] = P[-3,-3]*multiplier
# P[-2,-2] = P[-2,-2]*multiplier
# P[-1,-1] = P[-1,-1]*multiplier
# Q = np.eye(13) * 0.
# R = np.eye(2) * 0.0000000001

# P = np.eye(13) * 0.0000000001
# multiplier = 1000
# P[-3,-3] = P[-3,-3]*multiplier
# P[-2,-2] = P[-2,-2]*multiplier
# P[-1,-1] = P[-1,-1]*multiplier
# Q = np.eye(13) * 0.
# R = np.eye(2) * 0.00000001

system.initialize_EKF(P, Q, R)

system.solve()


# # point 1
# -0.5
# -0.3
# -1
# -1.1

# # point 2
# 1.15
# -0.2
# box:26
