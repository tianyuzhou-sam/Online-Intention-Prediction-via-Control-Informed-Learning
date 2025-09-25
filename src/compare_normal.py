import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as np
import math


iter = 100
goal = list()
loss = list()
data_time = list()
for i in range(iter):
    data = sio.loadmat('results/noise_0/result_' + str(i) + '.mat')
    
    if i == 0:
        goal = data['goal_error'][0]
        loss = data['Loss'][0]
        data_time = data['data_time'][0].tolist()
    elif not math.isnan(data['goal_error'][0][-1]):
        goal = np.vstack((goal, data['goal_error'][0]))
        loss = np.vstack((loss, data['Loss'][0]))
        data_time = np.hstack((data_time, data['data_time'][0].tolist()))
    # if math.isnan(data['goal_error'][0][-1]) or data['Loss'][0][-1] > 100:
    #     print(i)
    #     print(data['goal_error'][0][-1])


timestep = list(range(0, len(goal[0])))


goal_avg = np.mean(goal, 0)
goal_std = np.std(goal, 0)
goal_ub = goal_avg + 3*goal_std
goal_lb = goal_avg - 3*goal_std

loss_avg = np.mean(loss, 0)
loss_std = np.std(loss, 0)
loss_ub = loss_avg + 3*loss_std
loss_lb = loss_avg - 3*loss_std

goal02 = list()
loss02 = list()
data_time02 = list()
for i in range(iter):
    data = sio.loadmat('results/normal_02/result_' + str(i) + '.mat')
    
    if i == 0:
        goal02 = data['goal_error'][0]
        loss02 = data['Loss'][0]
        data_time02 = data['data_time'][0].tolist()
    elif not math.isnan(data['goal_error'][0][-1]):
        goal02 = np.vstack((goal02, data['goal_error'][0]))
        loss02 = np.vstack((loss02, data['Loss'][0]))
        data_time02 = np.hstack((data_time02, data['data_time'][0].tolist()))
    # if math.isnan(data['goal_error'][0][-1]) or data['Loss'][0][-1] > 100:
    #     print(i)
    #     print(data['goal_error'][0][-1])

goal_avg02 = np.mean(goal02, 0)
goal_std02 = np.std(goal02, 0)
goal_ub02 = goal_avg02 + 3*goal_std02
goal_lb02 = goal_avg02 - 3*goal_std02

loss_avg02 = np.mean(loss02, 0)
loss_std02 = np.std(loss02, 0)
loss_ub02 = loss_avg02 + 3*loss_std02
loss_lb02 = loss_avg02 - 3*loss_std02

goal05 = list()
loss05 = list()
data_time05 = list()
for i in range(iter):
    data = sio.loadmat('results/normal_05/result_' + str(i) + '.mat')
    
    if i == 0:
        goal05 = data['goal_error'][0]
        loss05 = data['Loss'][0]
        data_time05 = data['data_time'][0].tolist()
    elif not math.isnan(data['goal_error'][0][-1]):
        goal05 = np.vstack((goal05, data['goal_error'][0]))
        loss05 = np.vstack((loss05, data['Loss'][0]))
        data_time05 = np.hstack((data_time05, data['data_time'][0].tolist()))
    # if math.isnan(data['goal_error'][0][-1]) or data['Loss'][0][-1] > 100:
    #     print(i)
    #     print(data['goal_error'][0][-1])


goal_avg05 = np.mean(goal05, 0)
goal_std05 = np.std(goal05, 0)
goal_ub05 = goal_avg05 + 3*goal_std05
goal_lb05 = goal_avg05 - 3*goal_std05

loss_avg05 = np.mean(loss05, 0)
loss_std05 = np.std(loss05, 0)
loss_ub05 = loss_avg05 + 3*loss_std05
loss_lb05 = loss_avg05 - 3*loss_std05



plt.rcParams['font.size'] = 24
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["figure.figsize"] = (6,5)
fig, ax = plt.subplots()
line_goal05, = ax.plot(goal_avg05, color='r', linewidth=5)
ax.fill_between(timestep, goal_lb05, goal_ub05, color='lightcoral', alpha=0.7)
line_goal02, = ax.plot(goal_avg02, color='g', linewidth=5)
ax.fill_between(timestep, goal_lb02, goal_ub02, color='lightgreen')
line_goal, = ax.plot(goal_avg, color='b', linewidth=5)
ax.fill_between(timestep, goal_lb, goal_ub, color='lightskyblue')
ax.legend([line_goal, line_goal02, line_goal05], ['$\sigma=0$','$\sigma=0.2$','$\sigma=0.5$'])
# ax.set_title('Prediction error at different Gaussian noise level')
ax.set_xlabel('$t$')
ax.set_ylabel('Prediction Loss')
ax.set_ylim([0,140])
ax.set_xlim(0, 80)
# ax.set_yscale('log')

plt.tight_layout()
plt.show()

from matplotlib.ticker import MaxNLocator
plt.rcParams["figure.figsize"] = (8,8)
demo_state_noise = data['demo_state_noise']  # Shape: (101, 13)
demo_state = data['demo_state']  # Shape: (101, 13)
state = data['state'][-1]  # Shape: (99, 13) - predicted trajectory
t_time = data['data_time']

fig, ax = plt.subplots(3,1)
for idx in range(3):
    ax[idx].plot(demo_state[:, idx], 'r--', linewidth=5)
    ax[idx].plot(demo_state_noise[:, idx], 'g-', linewidth=5)
    ax[idx].plot(state[:, idx], 'b-', linewidth=8)
    ax[idx].plot(demo_state[:, idx], 'r--', linewidth=5)
    ax[idx].yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
ax[-1].set_xlabel('$t$')
ax[0].set_ylabel('$x$')
ax[1].set_ylabel('$y$')
ax[2].set_ylabel('$z$')
# remove x-axis tick labels on top two axes
ax[0].tick_params(labelbottom=False)
ax[1].tick_params(labelbottom=False)
ax[1].legend(['Ground Truth', 'Observation','Prediction'], loc='lower right', bbox_to_anchor=(1.1, 0.8))
# plt.tight_layout()
plt.show()