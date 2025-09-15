import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as np
import math
from casadi import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


iter = 99
goal = list()
loss = list()
for i in range(iter):
    data = sio.loadmat('results/nn/nn_result_' + str(i) + '.mat')
    if i == 0:
        goal = data['goal_error'][0]
        loss = data['Loss'][0]
    elif len(data['goal_error'][0]) == 801:    
        goal = np.vstack((goal, data['goal_error'][0]))
        loss = np.vstack((loss, data['Loss'][0]))
    # if i == 0:
    #     goal = data['goal_error'][0][:401]
    #     loss = data['Loss'][0][:401]
    # else:    
    #     goal = np.vstack((goal, data['goal_error'][0][:401]))
    #     loss = np.vstack((loss, data['Loss'][0][:401]))
    if math.isnan(data['goal_error'][0][-1]) or data['goal_error'][0][-1] > 10:
        print(i)
        print(data['goal_error'][0][-1])

timestep = list(range(0, len(goal[0])))


goal_avg = np.mean(goal, 0)
goal_std = np.std(goal, 0)
goal_ub = goal_avg + 3*goal_std
goal_lb = goal_avg - 3*goal_std

loss_avg = np.mean(loss, 0)
loss_std = np.std(loss, 0)
loss_ub = loss_avg + 3*loss_std
loss_lb = loss_avg - 3*loss_std

plt.rcParams['font.size'] = 30
plt.rcParams["figure.figsize"] = (10,8)
fig, ax = plt.subplots()
line_goal, = ax.plot(goal_avg, color='b', linewidth=4)
ax.fill_between(timestep, goal_lb, goal_ub, color='lightskyblue')
ax.set_ylim([0,250])
ax.set_xlabel('Number of data points')
ax.set_ylabel('Prediction error')
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_yscale('log')
# ax.set_title('Prediction error of goal state ')

fig, ax = plt.subplots()
line_loss, = ax.plot(loss_avg, color='b', linewidth=4)
ax.fill_between(timestep, loss_lb, loss_ub, color='lightskyblue')
axins1 = zoomed_inset_axes(ax, 5, loc=7)
axins1.plot(timestep[80:150], loss_avg[80:150], color="b", linewidth=3)
axins1.fill_between(timestep[80:150], loss_lb[80:150], loss_ub[80:150], color='lightskyblue')
axins1.set_ylim([1e1,4000])
axins1.set_xlim(80, 150)
mark_inset(ax, axins1, loc1=2, loc2=3, fc="none", ec="0.5")
# ax.set_ylim([0,35000])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.set_ylim([1e1,80000])
ax.set_xlabel('Number of data points')
ax.set_ylabel('Trajectory Loss')
# ax.set_yscale('log')
# ax.set_title('Trajectory loss')

plt.show()






# data = sio.loadmat('results/nn/old/result_0.mat')
# theta_error = data['theta_error']
# goal_error = data['goal_error'][0]
# true_theta = data['true_theta'][0]
# theta_his = data['theta']
# demo_state_traj_original = data['demo_state']
# demo_state_traj = data['demo_state_noise']
# demo_control_traj = data['demo_control']
# x_his = data['state']
# u_his = data['control']
# data_time = data['data_time'][0]
# gradient_time = data['gradient_time'][0]
# ekf_time = data['ekf_time'][0]


# Loss_his = []
# for i in range(801):
#     lossnorm = 0
#     for j in range(80):
#         lossnorm += norm_2(x_his[i][j]-demo_state_traj_original[j])
#     Loss_his += [np.asarray(lossnorm)[0,0]]


# sio.savemat("results/nn_result_0.mat", {'Loss': Loss_his,
#                                             'theta_error': theta_error, 'goal_error': goal_error,
#                                             'true_theta': true_theta, 'theta': theta_his,
#                                             'demo_state': demo_state_traj_original,
#                                             'demo_state_noise': demo_state_traj, 'demo_control': demo_control_traj,
#                                             'state': x_his, 'control': u_his,
#                                             'data_time': data_time, 'gradient_time': gradient_time,
#                                             'ekf_time': ekf_time})

