import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
import transforms3d
import time
import matplotlib.pyplot as plt 
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
from PDP import PDP
sys.path.append(os.getcwd() + '/src')
import Env


init_position = [0,0,0]
init_velocity = [0,0,0]

goal_position = [5, 5, 0]


goal_v_I = np.array([0, 0, 0])
goal_q = Env.toQuaternion(0, [0, 0, 1])
goal_w_B = np.array([0, 0, 0])

true_theta = np.hstack([1, 1, 1, 1, 0.4, 10, 1, 5, 1, goal_position, goal_v_I, goal_q, goal_w_B])
wthrust=0.1

dt = 0.15
horizon = 80

x0 = np.hstack([init_position, init_velocity, goal_q, goal_w_B])
xf = np.hstack([goal_position, goal_v_I, goal_q, goal_w_B])

# ------------------------------ Set up dynamic system ------------------------------
nnFactor = 1

dynsys = Env.Quadrotor()
dynsys.initDyn()
n_state = dynsys.X.size()[0]
n_control = dynsys.U.size()[0]
# dynsys.initNeuralDyn(hidden_layers=[nnFactor*(n_state+n_control), nnFactor*(n_state+n_control)])
dynsys.initDyn(true_theta[0], true_theta[1], true_theta[2], true_theta[3], true_theta[4])
dynsys.initCost(true_theta[5], true_theta[6], true_theta[7], true_theta[8], goal_position, goal_v_I, goal_q, goal_w_B, wthrust)

theta = sio.loadmat('theta.mat')
theta = theta['theta'][0]
# theta = np.random.random((1,846)).flatten()
# theta = []

R = np.array([[1,0,0],[0,1,0],[0,0,1]]) # rotation matrix in numpy 2D array
init_state = np.hstack([init_position, init_velocity, transforms3d.quaternions.mat2quat(R).tolist(), 0, 0, 0])

demoSys = PDP.OCSys()
demoSys.setAuxvarVariable(vertcat(dynsys.dyn_auxvar, dynsys.cost_auxvar))
demoSys.setControlVariable(dynsys.U)
demoSys.setStateVariable(dynsys.X)
truedyn = dynsys.X + dt * dynsys.f
demoSys.setDyn(truedyn)
demoSys.setPathCost(dynsys.path_cost)
demoSys.setFinalCost(dynsys.final_cost)
demoSys.diffPMP()


Qk = np.zeros((n_state, n_state))
weight = np.array([true_theta[5],true_theta[5],true_theta[5],true_theta[6],true_theta[6],true_theta[6],true_theta[7],true_theta[7],true_theta[7],true_theta[7],true_theta[8],true_theta[8],true_theta[8]])
# weight = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
Qf = np.diag(weight.flatten())*2*1e0
weight = np.array([wthrust,wthrust,wthrust,wthrust])
Rk = np.diag(weight.flatten())*2*1e0

gamma = 0.001

A = np.zeros((horizon,n_state,n_state))
B = np.zeros((horizon,n_state,n_control))
S = np.zeros((horizon+1,n_state,n_state))
K = np.zeros((horizon,n_control,n_state))
Kv = np.zeros((horizon,n_control,n_state))
v = np.zeros((horizon+1,n_state,1))
Ku = np.zeros((horizon,n_control,n_control))

ref_state = np.zeros((horizon+1,n_state))
ref_control = np.zeros((horizon,n_control))

eq = [0,0,0,0]
# eq = [2.5,2.5,2.5,2.5]

ref_state[0] = x0
for idx in range(horizon):
    ref_control[idx] = [0,0,0,0]
    for state in range(n_state):
        ref_state[idx+1][state] = init_state


fig, axs = plt.subplots(3,1)
for idx in range(3):
    axs[idx].plot(ref_state[:,idx],'b')
    axs[idx].plot(80, xf[idx],'r*')
axs[0].set_ylabel("$x$")
axs[1].set_ylabel("$y$")
axs[2].set_ylabel("$z$")
axs[-1].set_xlabel("$t$")
axs[0].set_title("State Trajectory")
# axs[0].legend(['Predicted State','Observed State','True State'])
plt.show()

start_time = time.time()
for iter in range(500):
    for idx in range(horizon):
        xk = ref_state[idx,:]
        uk = ref_control[idx,:]+eq
        A[idx] = demoSys.dfx_fn(xk, uk, theta)
        B[idx] = demoSys.dfu_fn(xk, uk, theta)

    S[horizon] = Qf

    v[horizon] = np.matmul(Qf, (ref_state[-1] - xf).reshape((n_state,1)))


    for idx in range(horizon-1,-1,-1):
        K[idx] = np.matmul(np.linalg.inv(np.matmul(np.transpose(B[idx]), np.matmul(S[idx+1], B[idx])) + Rk), np.matmul(np.transpose(B[idx]), np.matmul(S[idx+1], A[idx])))
        S[idx] = np.matmul(np.matmul(np.transpose(A[idx]), S[idx+1]), (A[idx]-np.matmul(B[idx], K[idx]))) + Qk
        v[idx] =  np.matmul(np.transpose(A[idx]-np.matmul(B[idx], K[idx])), v[idx+1]) - np.matmul(np.transpose(K[idx]), np.matmul(Rk, ref_control[idx].reshape(n_control,1))) 
        Kv[idx] = np.matmul(np.linalg.inv(np.matmul(np.transpose(B[idx]), np.matmul(S[idx+1], B[idx])) + Rk), np.transpose(B[idx]))
        Ku[idx] = np.matmul(np.linalg.inv(np.matmul(np.transpose(B[idx]), np.matmul(S[idx+1], B[idx])) + Rk), Rk)

    control_traj = np.zeros((horizon, n_control))
    state_traj = np.zeros((horizon+1, n_state))

    state_traj[idx] = init_state
    for idx in range(horizon):
        control_traj[idx] = ref_control[idx] - gamma*(np.matmul(K[idx], (state_traj[idx] - ref_state[idx]).reshape(n_state,1)) + np.matmul(Kv[idx], v[idx+1]) + np.matmul(Ku[idx], ref_control[idx].reshape(n_control,1))).flatten()
        ref_control[idx] = control_traj[idx]
        control_traj[idx] += eq
        # for jdx in range(len(control_traj[idx])):
            # if control_traj[idx][jdx] > 3:
            #     control_traj[idx][jdx] = 3
            # if control_traj[idx][jdx] < 0:
            #     control_traj[idx][jdx] = 0
        for state in range(n_state):
            state_traj[idx+1][state] = demoSys.dyn_fn(state_traj[idx], control_traj[idx], theta)[state]
            # state_traj[idx+1][state] = demoSys.dyn_fn(state_traj[idx], ref_control[idx], theta)[state]

    ref_state = state_traj
    # ref_control = control_traj

    obj = 0
    for idx in range(horizon):
        obj += 0.5*np.matmul(control_traj[idx], np.matmul(Rk, np.transpose(ref_control[idx])))
    dgoal = state_traj[-1]-xf
    obj += 0.5*np.matmul(dgoal, np.matmul(Qf, np.transpose(dgoal)))

    print(obj)

    # print(ref_control)
    # print(state_traj)
    print(state_traj[-1])
    print(norm_2(state_traj[-1][:3]-xf[:3]))
    print(norm_2(state_traj[-1]-xf))

    print(time.time()-start_time)

fig, axs = plt.subplots(3,1)
for idx in range(3):
    axs[idx].plot(state_traj[:,idx],'b')
    axs[idx].plot(80, xf[idx],'r*')
axs[0].set_ylabel("$x$")
axs[1].set_ylabel("$y$")
axs[2].set_ylabel("$z$")
axs[-1].set_xlabel("$t$")
axs[0].set_title("State Trajectory")
# axs[0].legend(['Predicted State','Observed State','True State'])
plt.show()

