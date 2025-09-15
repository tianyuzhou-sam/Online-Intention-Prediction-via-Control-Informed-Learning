import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
import transforms3d
import matplotlib.pyplot as plt 
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
from PDP import PDP
sys.path.append(os.getcwd() + '/src')
from EKF import EKF
import ImitationLearning
import Env
# numerical difference
# initial guess
# max iter
for iter in range(0, 1):
    # ------------------------------ Set up parameters ------------------------------
    init_position = [0,0,0]
    init_velocity = [2*(np.random.random()-0.5),2*(np.random.random()-0.5),2*(np.random.random()-0.5)]
    init_velocity = [0,0,0]

    goal_range = 10
    goal_offset = 2
    height_range = 2
    angle = (np.random.random()-0.5) * 2 * np.pi
    goal_position = [goal_range*np.cos(angle)+(np.random.random()-0.5)*goal_offset, goal_range*np.sin(angle)+(np.random.random()-0.5)*goal_offset, height_range*np.random.random()]


    goal_v_I = np.array([0, 0, 0])
    goal_q = Env.toQuaternion(0, [0, 0, 1])
    goal_w_B = np.array([0, 0, 0])
    true_theta = np.hstack([1, 1, 1, 1, 0.4, 0.01, 10, 1, 5, 1, goal_position, goal_v_I, goal_q, goal_w_B])

    dt = 0.15
    horizon = 80
    noise = 0.5

    init_range = 10
    angle = (np.random.random()-0.5) * 2 * np.pi
    dist = np.random.random()*init_range
    pred = [goal_position[0]+np.cos(angle)*dist,goal_position[1]+np.sin(angle)*dist,goal_position[2]+(np.random.random()-0.5)*height_range]
    
    pred = init_position
    pred_init = np.hstack([pred, init_velocity, goal_q, goal_w_B])

    # ------------------------------ Set up dynamic system ------------------------------
    project = str(iter)
    saveFlag = False
    nnFactor = 1

    dynsys = Env.Quadrotor()
    dynsys.initDyn()
    n_state = dynsys.X.size()[0]
    n_control = dynsys.U.size()[0]
    # sysid.n_auxvar is 3889
    # dynsys.initNeuralDyn(hidden_layers=[2*(n_state+n_control), 4*(n_state+n_control)])
    # nn 3 5793
    # dynsys.initNeuralDyn(hidden_layers=[2*(n_state+n_control), 4*(n_state+n_control), 2*(n_state+n_control)])
    # nn 1 2121
    dynsys.initNeuralDyn(hidden_layers=[4*(n_state+n_control)])
    dynsys.initCost(wthrust=0.1)

    trueSys = Env.Quadrotor()
    trueSys.initDyn(true_theta[0], true_theta[1], true_theta[2], true_theta[3], true_theta[4], true_theta[5])
    trueSys.initCost(0.1, true_theta[6], true_theta[7], true_theta[8], true_theta[9], goal_position, goal_v_I, goal_q, goal_w_B)

    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
    init_state = np.hstack([init_position, init_velocity, transforms3d.quaternions.mat2quat(R).tolist(), 0, 0, 0])

    system = ImitationLearning.ImitationLearningNN(project, init_state, true_theta, dynsys, trueSys, dt, horizon, noise, pred_init, saveFlag)
    demo_traj = system.generate_traj()
    demo_state_traj = demo_traj['state_traj_opt']
    demo_control_traj = demo_traj['control_traj_opt']

    sysid = PDP.SysID()
    sysid.setAuxvarVariable(dynsys.dyn_auxvar)
    sysid.setControlVariable(dynsys.U)
    sysid.setStateVariable(dynsys.X)
    # random initial guess
    theta = np.random.random(sysid.n_auxvar)-0.5


    system.set_iteration(1)
    system.initialize_nn_parameter(theta)
    
    # --------------------------- initilize EKF ----------------------------------------
    # sysid.n_auxvar is 3889
    print(sysid.n_auxvar)
    P = np.eye(sysid.n_auxvar+4+13) * 0.00000001
    for idx in range(sysid.n_auxvar,sysid.n_auxvar+4):
        P[idx,idx] = P[idx,idx]*1
    for idx in range(sysid.n_auxvar+4,sysid.n_auxvar+7):
        P[idx,idx] = P[idx,idx]*1000
    for idx in range(sysid.n_auxvar+7,sysid.n_auxvar+17):
        P[idx,idx] = P[idx,idx]*1
    Q = np.eye(sysid.n_auxvar+4+13) * 0.
    R = np.eye(13) * 0.00000001

    system.initialize_EKF(P, Q, R)
    system.set_iteration(5)

    system.solve()
    # print('case ' + str(iter) + ' done')



##########################
    # normal 0.5
    # P = np.eye(sysid.n_auxvar+4+13) * 0.00000001
    # for idx in range(sysid.n_auxvar,sysid.n_auxvar+4):
    #     P[idx,idx] = P[idx,idx]*1
    # for idx in range(sysid.n_auxvar+4,sysid.n_auxvar+7):
    #     P[idx,idx] = P[idx,idx]*1000
    # for idx in range(sysid.n_auxvar+7,sysid.n_auxvar+17):
    #     P[idx,idx] = P[idx,idx]*1
    # Q = np.eye(sysid.n_auxvar+4+13) * 0.
    # R = np.eye(13) * 0.00000001
