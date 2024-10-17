import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as np
import math


iter = 100
goal = list()
loss = list()
for i in range(iter):
    data = sio.loadmat('results/result_' + str(i) + '.mat')
    if i == 0:
        goal = data['goal_error'][0]
        loss = data['Loss'][0]
    elif not math.isnan(data['goal_error'][0][-1]):
        goal = np.vstack((goal, data['goal_error'][0]))
        loss = np.vstack((loss, data['Loss'][0]))
    if math.isnan(data['goal_error'][0][-1]) or data['goal_error'][0][-1] > 1:
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

fig, ax = plt.subplots()
line_goal, = ax.plot(goal_avg, color='b', linewidth=4)
ax.fill_between(timestep, goal_lb, goal_ub, color='lightskyblue')
ax.set_ylim([0,60])
# ax.set_yscale('log')

fig, ax = plt.subplots()
line_loss, = ax.plot(loss_avg, color='b', linewidth=4)
ax.fill_between(timestep, loss_lb, loss_ub, color='lightskyblue')
# ax.set_ylim([0,1000])
# ax.set_yscale('log')

plt.show()