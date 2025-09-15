import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
import transforms3d
sys.path.append(os.getcwd() + '/experiment')
import ImitationLearning
import Env


case = 'dog'


# ------------------------------ Set up parameters ------------------------------
if case == 'dog':
    init_state = [-2.86,-0.58,-1.00]
    # wx = 100
    # wd = 0.07
    # wu1 = 1
    # wu2 = 3
    # wu3 = 1
    wx = 120
    wd = 0.06
    wu1 = 1
    wu2 = 2.5
    wu3 = 1
    true_theta = np.hstack([wx, wd, wu1, wu2, wu3])

    dt = 0.1
    horizon = 100

    pred_init = init_state[:2]

    goal = [2.728,-0.827]

    goal_v_I = [0,0]

    # ------------------------------ Set up dynamic system ------------------------------
    project = str(case)
    saveFlag = False
    dynsys = Env.Dog()
    dynsys.initDyn()
    dynsys.initCost()
                
    system = ImitationLearning.ImitationLearning(project, init_state, true_theta, dynsys, dt, horizon, pred_init, saveFlag)
    system.set_iteration(1)

    # --------------------------- initilize EKF ----------------------------------------
    P = np.eye(7) * 0.00000001
    multiplier = 1000
    P[5,5] = P[5,5]*multiplier
    P[6,6] = P[6,6]*multiplier
    Q = np.eye(7) * 0.
    R = np.eye(2) * 0.000000001

    system.initialize_EKF(P, Q, R)

    system.solve()

if case == 3:
    init_state = [-2.219,-0.980,-0.923]
    wx = 100
    wv = 1
    wd = 0.1
    wd2 = 0.1
    wu1 = 1
    wu2 = 3
    wu3 = 1
    true_theta = np.hstack([wx, wv, wd, wd2, wu1, wu2, wu3])

    dt = 0.1
    horizon = 165

    pred_init = init_state[:2]

    goal = [1.736,0.456]

    goal_v_I = [0,0]

    # ------------------------------ Set up dynamic system ------------------------------
    project = str(case)
    saveFlag = False
    dynsys = Env.Dog()
    dynsys.initDyn()
    dynsys.initCost2()
                
    system = ImitationLearning.ImitationLearning(project, init_state, true_theta, dynsys, dt, horizon, pred_init, saveFlag)
    system.set_iteration(1)

    # --------------------------- initilize EKF ----------------------------------------
    P = np.eye(9) * 0.00000001
    multiplier = 1000
    P[-2,-2] = P[-2,-2]*multiplier
    P[-1,-1] = P[-1,-1]*multiplier
    Q = np.eye(9) * 0.
    R = np.eye(2) * 0.000000001

    system.initialize_EKF(P, Q, R)

    system.solve()

if case == 4:
    init_state = [-2.219,-0.980,-0.923]
    # wx = 100
    # wv = 1
    # wd = 0.07
    # wu1 = 1
    # wu2 = 3
    # wu3 = 1
    # true_theta = np.hstack([wx, wv, wd, wu1, wu2, wu3])



    dt = 0.1
    horizon = 165

    pred_init = init_state[:2]

    goal = [1.736,0.456,-1.09]


    # ------------------------------ Set up dynamic system ------------------------------
    project = str(case)
    saveFlag = False
    dynsys = Env.Dog()
    dynsys.initDyn()
    dynsys.initCostNN(hidden_layers=[128])

    true_theta = np.random.random(dynsys.n_cost_auxvar) - 0.5
                
    system = ImitationLearning.ImitationLearning(project, init_state, true_theta, dynsys, dt, horizon, pred_init, saveFlag)
    system.set_iteration(1)

    # --------------------------- initilize EKF ----------------------------------------
    P = np.eye(1027) * 0.00000001
    multiplier = 1000
    P[-1,-1] = P[-1,-1]*multiplier
    P[-2,-2] = P[-2,-2]*multiplier
    Q = np.eye(1027) * 0.
    R = np.eye(2) * 0.000000001

    system.initialize_EKF(P, Q, R)
    system.set_iteration(10)

    system.solve()

elif case == 5:
    init_state = [-2.676,-0.869,-1.00]
    true_theta = [1,0,0,0,0,0,1,0,0]

    dt = 0.1
    horizon = 140

    pred_init = init_state[:2]

    goal = [2.728,-0.827]


    # ------------------------------ Set up dynamic system ------------------------------
    project = str(case)
    saveFlag = False
    dynsys = Env.Dog()
    dynsys.initDyn()
    dynsys.initCostPoly()

    true_theta[-2] = goal[0]
    true_theta[-1] = goal[1]
    system = ImitationLearning.ImitationLearning(project, init_state, true_theta, dynsys, dt, horizon, pred_init, saveFlag)
    system.set_iteration(1)

    # --------------------------- initilize EKF ----------------------------------------
    P = np.eye(9) * 0.0000000001
    multiplier = 1000
    P[-1,-1] = P[-1,-1]*multiplier
    P[-2,-2] = P[-2,-2]*multiplier
    Q = np.eye(9) * 0.
    R = np.eye(2) * 0.000000001

    system.initialize_EKF(P, Q, R)

    system.solve()

