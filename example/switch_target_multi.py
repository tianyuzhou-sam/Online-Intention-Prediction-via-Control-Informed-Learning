import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
import transforms3d
sys.path.append(os.getcwd() + '/src')
import ImitationLearningSwitch
import Env

for iter in range(100):
    # ------------------------------ Set up parameters ------------------------------
    init_position = [0,0,0]
    # init_velocity = [2*(np.random.random()-0.5),2*(np.random.random()-0.5),2*(np.random.random()-0.5)]
    init_velocity = [0,0,0]

    goal_range = 5
    angle = (np.random.random()-0.5) * 2 * np.pi
    goal_position = [goal_range*np.cos(angle), goal_range*np.sin(angle), 1]

    
    goal_v_I = np.array([0, 0, 0])
    goal_q = Env.toQuaternion(0, [0, 0, 1])
    goal_w_B = np.array([0, 0, 0])

    true_theta = np.hstack([1, 1, 1, 1, 0.4, 0.01, 0.1, 100, 1, 5, 1, goal_position, goal_v_I, goal_q, goal_w_B])

    dt = 0.15
    noise = 0.
    horizon = 120
    H = 120
    MemoryTime = 10

    pred = init_position
    pred_init = np.hstack([pred, goal_v_I, goal_q, goal_w_B])

    # ------------------------------ Set up dynamic system ------------------------------
    project = str(iter)
    saveFlag = False
    window = 60
    dynsys = Env.Quadrotor()
    dynsys.initDyn()
    dynsys.initCost()

    trueSys = Env.Quadrotor()
    trueSys.initDyn(true_theta[0], true_theta[1], true_theta[2], true_theta[3], true_theta[4], true_theta[5])
    trueSys.initCost(true_theta[6], true_theta[7], true_theta[8], true_theta[9], true_theta[10], goal_position, goal_v_I, goal_q, goal_w_B)

    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
    init_state = np.hstack([init_position, init_velocity, transforms3d.quaternions.mat2quat(R).tolist(), 0, 0, 0])

    system = ImitationLearningSwitch.ImitationLearning(project, init_state, true_theta, dynsys, trueSys, dt, horizon, H, MemoryTime, window, noise, pred_init, saveFlag)
    system.set_iteration(1)
    system.set_sigma(0.)
    system.initialize_parameter()

    system.set_goal_range(goal_range)




    # dt = 60
    # if dt == 60:
    #     switch_time = [30,30+dt]
    # if dt == 50:
    #     switch_time = [30,30+dt]
    # if dt == 40:
    #     switch_time = [30,30+dt]
    # if dt == 30:
    #     switch_time = [50,50+dt]
    # if dt == 20:
    #     switch_time = [60,60+dt]
    # if dt == 10:
    #     switch_time = [70,70+dt]
    # goal_position1 = [5,5,1]
    # goal_position1[0] = goal_position1[0] + np.random.random()-0.5
    # goal_position1[1] = goal_position1[1] + np.random.random()-0.5
    # goal_position2 = [5,5,1]
    # switch_goal = [goal_position1, goal_position2]
    # system.switch_target(switch_time, switch_goal)

    # --------------------------- initilize EKF ----------------------------------------
    P = np.eye(24) * 0.00000001
    # P = np.eye(24) * 0.001
    for idx in range(0,11):
        P[idx,idx] = P[idx,idx]*1
    for idx in range(11,14):
        P[idx,idx] = P[idx,idx]*1000*5
    Q = np.eye(24) * 0.
    for idx in range(11,14):
        Q[idx,idx] = 0.0000000
    # R = np.eye(13) * 0.001
    R = np.eye(13) * 0.0000001



    system.initialize_EKF(P, Q, R)

    system.solve()
    # print('case ' + str(iter) + ' done')



##########################
    # P = np.eye(24) * 0.0000001
    # # P = np.eye(24) * 0.001
    # for idx in range(0,11):
    #     P[idx,idx] = P[idx,idx]*1
    # for idx in range(11,14):
    #     P[idx,idx] = P[idx,idx]*1000
    # Q = np.eye(24) * 0.
    # # R = np.eye(13) * 0.001
    # R = np.eye(13) * 0.0000001
    
    # for 0 noise
    # P = np.eye(24) * 0.0000001
    # for idx in range(11,14):
    #     P[idx,idx] = P[idx,idx]*1000
    # Q = np.eye(24) * 0.
    # R = np.eye(13) * 0.000000001

    # for 0.1 uniform
    # for 0.1 normal
    # P = np.eye(24) * 0.00000001
    # for idx in range(11,14):
    #     P[idx,idx] = P[idx,idx]*10000
    # Q = np.eye(24) * 0.
    # R = np.eye(13) * 0.0000001

    # for 0.5 uniform
    # P = np.eye(24) * 0.000000001
    # for idx in range(11,14):
    #     P[idx,idx] = P[idx,idx]*10000
    # Q = np.eye(24) * 0.
    # R = np.eye(13) * 0.0000001

    # for 1 uniform
    # for 1 normal
    # P = np.eye(24) * 0.00000000001
    # for idx in range(11,14):
    #     P[idx,idx] = P[idx,idx]*100000
    # Q = np.eye(24) * 0.
    # R = np.eye(13) * 0.0000001

    # for 0.5 normal
    # P = np.eye(24) * 0.000000001
    # for idx in range(11,14):
    #     P[idx,idx] = P[idx,idx]*10000
    # Q = np.eye(24) * 0.
    # R = np.eye(13) * 0.0000001

    # for 0.2 normal normal
    # P = np.eye(24) * 0.000000001
    # for idx in range(11,14):
    #     P[idx,idx] = P[idx,idx]*1000
    # Q = np.eye(24) * 0.
    # R = np.eye(13) * 0.0000001

