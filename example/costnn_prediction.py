import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
import transforms3d
sys.path.append(os.getcwd() + '/src')
import ImitationLearning
import Env

for iter in range(0, 1):
    # ------------------------------ Set up parameters ------------------------------
    init_position = [0,0,0]
    init_velocity = [2*(np.random.random()-0.5),2*(np.random.random()-0.5),2*(np.random.random()-0.5)]

    goal_range = 10
    goal_offset = 2
    height_range = 2
    angle = (np.random.random()-0.5) * 2 * np.pi
    goal_position = [goal_range*np.cos(angle)+(np.random.random()-0.5)*goal_offset, goal_range*np.sin(angle)+(np.random.random()-0.5)*goal_offset, height_range*np.random.random()]

    goal_v_I = np.array([0, 0, 0])
    goal_q = Env.toQuaternion(0, [0, 0, 1])
    goal_w_B = np.array([0, 0, 0])
    true_theta = np.hstack([1, 1, 1, 1, 0.4, 10, 1, 5, 1, goal_position, goal_v_I, goal_q, goal_w_B])
    
    dt = 0.15
    horizon = 80
    noise = 0

    init_range = 10
    angle = (np.random.random()-0.5) * 2 * np.pi
    dist = np.random.random()*init_range
    pred = [goal_position[0]+np.cos(angle)*dist,goal_position[1]+np.sin(angle)*dist,goal_position[2]+(np.random.random()-0.5)*height_range]

    pred_init = np.hstack([pred, goal_v_I, goal_q, goal_w_B])

    # ------------------------------ Set up dynamic system ------------------------------
    project = str(iter)
    saveFlag = True
    dynsys = Env.Quadrotor()
    dynsys.initDyn(true_theta[0], true_theta[1], true_theta[2], true_theta[3], true_theta[4],c=0.01)
    n_state = dynsys.X.size()[0]
    n_control = dynsys.U.size()[0]
    dynsys.initNeuralCost(hidden_layers=[1*(n_state+n_control), 1*(n_state+n_control)])

    trueSys = Env.Quadrotor()
    trueSys.initDyn(true_theta[0], true_theta[1], true_theta[2], true_theta[3], true_theta[4])
    trueSys.initCost(true_theta[5], true_theta[6], true_theta[7], true_theta[8], goal_position, goal_v_I, goal_q, goal_w_B)

    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
    init_state = np.hstack([init_position, init_velocity, transforms3d.quaternions.mat2quat(R).tolist(), 0, 0, 0])

    system = ImitationLearning.ImitationLearningNN(project, init_state, true_theta, dynsys, trueSys, dt, horizon, noise, pred_init, saveFlag)
    # system.set_sigma(0.5)
    system.set_iteration(1)
    # system.initialize_parameter()

    # --------------------------- initilize EKF ----------------------------------------
    P = np.eye(643) * 0.001
    # for idx in range(9,12):
    #     P[idx,idx] = P[idx,idx]*10000
    # for idx in range(12,22):
    #     P[idx,idx] = P[idx,idx]*0
    Q = np.eye(643) * 0.
    R = np.eye(13) * 0.000001

    system.initialize_EKF(P, Q, R)

    system.solve()
    # print('case ' + str(iter) + ' done')



##########################
    # for 0 noise
    # P = np.eye(22) * 0.0000001
    # for idx in range(9,12):
    #     P[idx,idx] = P[idx,idx]*1000
    # # for idx in range(12,22):
    # #     P[idx,idx] = P[idx,idx]*0
    # Q = np.eye(22) * 0.
    # R = np.eye(13) * 0.000000001

    # for 0.1 noise
    # for normal 01
    # P = np.eye(22) * 0.00000001
    # for idx in range(9,12):
    #     P[idx,idx] = P[idx,idx]*10000
    # # for idx in range(12,22):
    # #     P[idx,idx] = P[idx,idx]*0
    # Q = np.eye(22) * 0.
    # R = np.eye(13) * 0.0000001

    # for 1 noise
    # P = np.eye(22) * 0.000000001
    # for idx in range(9,12):
    #     P[idx,idx] = P[idx,idx]*10000
    # # for idx in range(12,22):
    # #     P[idx,idx] = P[idx,idx]*0
    # Q = np.eye(22) * 0.
    # R = np.eye(13) * 0.0000001

    # for 2 noise
    # ofr 1 normal
    # P = np.eye(22) * 0.00000000001
    # for idx in range(9,12):
    #     P[idx,idx] = P[idx,idx]*100000
    # # for idx in range(12,22):
    # #     P[idx,idx] = P[idx,idx]*0
    # Q = np.eye(22) * 0.
    # R = np.eye(13) * 0.0000001

    # normal 0.5
    # P = np.eye(22) * 0.000000001
    # for idx in range(9,12):
    #     P[idx,idx] = P[idx,idx]*1000
    # Q = np.eye(22) * 0.
    # R = np.eye(13) * 0.0000001

    # normal 0.2
    # P = np.eye(22) * 0.000000001
    # for idx in range(9,12):
    #     P[idx,idx] = P[idx,idx]*1000
    # Q = np.eye(22) * 0.
    # R = np.eye(13) * 0.0000001