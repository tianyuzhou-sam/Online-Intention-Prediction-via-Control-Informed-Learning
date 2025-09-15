import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as np
import math


iter = 1000
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

goal01 = list()
loss01 = list()
data_time01 = list()
for i in range(iter):
    data = sio.loadmat('results/uniform_01/result_' + str(i) + '.mat')
    
    if i == 0:
        goal01 = data['goal_error'][0]
        loss01 = data['Loss'][0]
        data_time01 = data['data_time'][0].tolist()
    elif not math.isnan(data['goal_error'][0][-1]):
        goal01 = np.vstack((goal01, data['goal_error'][0]))
        loss01 = np.vstack((loss01, data['Loss'][0]))
        data_time01 = np.hstack((data_time01, data['data_time'][0].tolist()))
    # if math.isnan(data['goal_error'][0][-1]) or data['Loss'][0][-1] > 100:
    #     print(i)
    #     print(data['goal_error'][0][-1])

goal_avg01 = np.mean(goal01, 0)
goal_std01 = np.std(goal01, 0)
goal_ub01 = goal_avg01 + 3*goal_std01
goal_lb01 = goal_avg01 - 3*goal_std01

loss_avg01 = np.mean(loss01, 0)
loss_std01 = np.std(loss01, 0)
loss_ub01 = loss_avg01 + 3*loss_std01
loss_lb01 = loss_avg01 - 3*loss_std01

goal1 = list()
loss1 = list()
data_time1 = list()
for i in range(iter):
    data = sio.loadmat('results/uniform_05/result_' + str(i) + '.mat')
    
    if i == 0:
        goal1 = data['goal_error'][0]
        loss1 = data['Loss'][0]
        data_time1 = data['data_time'][0].tolist()
    elif not math.isnan(data['goal_error'][0][-1]):
        goal1 = np.vstack((goal1, data['goal_error'][0]))
        loss1 = np.vstack((loss1, data['Loss'][0]))
        data_time1 = np.hstack((data_time1, data['data_time'][0].tolist()))
    # if math.isnan(data['goal_error'][0][-1]) or data['Loss'][0][-1] > 100:
    #     print(i)
    #     print(data['goal_error'][0][-1])


goal_avg1 = np.mean(goal1, 0)
goal_std1 = np.std(goal1, 0)
goal_ub1 = goal_avg1 + 3*goal_std1
goal_lb1 = goal_avg1 - 3*goal_std1

loss_avg1 = np.mean(loss1, 0)
loss_std1 = np.std(loss1, 0)
loss_ub1 = loss_avg1 + 3*loss_std1
loss_lb1 = loss_avg1 - 3*loss_std1

goal2 = list()
loss2 = list()
data_time2 = list()
for i in range(iter):
    data = sio.loadmat('results/uniform_1/result_' + str(i) + '.mat')
    
    if i == 0:
        goal2 = data['goal_error'][0]
        loss2 = data['Loss'][0]
        data_time2 = data['data_time'][0].tolist()
    elif not math.isnan(data['goal_error'][0][-1]):
        goal2 = np.vstack((goal2, data['goal_error'][0]))
        loss2 = np.vstack((loss2, data['Loss'][0]))
        data_time2 = np.hstack((data_time2, data['data_time'][0].tolist()))
    # if math.isnan(data['goal_error'][0][-1]) or data['Loss'][0][-1] > 1000:
    #     print(i)
    #     print(data['goal_error'][0][-1])


goal_avg2 = np.mean(goal2, 0)
goal_std2 = np.std(goal2, 0)
goal_ub2 = goal_avg2 + 3*goal_std2
goal_lb2 = goal_avg2 - 3*goal_std2

loss_avg2 = np.mean(loss2, 0)
loss_std2 = np.std(loss2, 0)
loss_ub2 = loss_avg2 + 3*loss_std2
loss_lb2 = loss_avg2 - 3*loss_std2

plt.rcParams['font.size'] = 24
plt.rcParams["figure.figsize"] = (10,8)
fig, ax = plt.subplots()
line_goal2, = ax.plot(goal_avg2, color='purple', linewidth=4)
ax.fill_between(timestep, goal_lb2, goal_ub2, color='violet')
line_goal1, = ax.plot(goal_avg1, color='r', linewidth=4)
ax.fill_between(timestep, goal_lb1, goal_ub1, color='lightcoral', alpha=0.7)
line_goal01, = ax.plot(goal_avg01, color='g', linewidth=4)
ax.fill_between(timestep, goal_lb01, goal_ub01, color='lightgreen')
line_goal, = ax.plot(goal_avg, color='b', linewidth=4)
ax.fill_between(timestep, goal_lb, goal_ub, color='lightskyblue')
ax.legend([line_goal, line_goal01, line_goal1, line_goal2], ['$\sigma=0$','$\sigma=0.1$','$\sigma=0.5$','$\sigma=1$'])
# ax.set_title('Prediction error at different uniform noise level')
ax.set_xlabel('$t$')
ax.set_ylabel('Prediction error')
ax.set_ylim([0,150])
# ax.set_yscale('log')

fig, ax = plt.subplots()
line_loss2, = ax.plot(loss_avg2, color='purple', linewidth=4)
ax.fill_between(timestep, loss_lb2, loss_ub2, color='violet')
line_loss1, = ax.plot(loss_avg1, color='r', linewidth=4)
ax.fill_between(timestep, loss_lb1, loss_ub1, color='lightcoral', alpha=0.7)
line_loss01, = ax.plot(loss_avg01, color='g', linewidth=4)
ax.fill_between(timestep, loss_lb01, loss_ub01, color='lightgreen')
line_loss, = ax.plot(loss_avg, color='b', linewidth=4)
ax.fill_between(timestep, loss_lb, loss_ub, color='lightskyblue')
ax.legend([line_loss, line_loss01, line_loss1, line_loss2], ['$\sigma=0$','$\sigma=0.1$','$\sigma=0.5$','$\sigma=1$'])
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.set_ylim([0,4000])
# ax.set_title('Trajectory loss at different uniform noise level')
ax.set_xlabel('$t$')
ax.set_ylabel('Trajectory Loss')
# ax.set_yscale('log')

plt.show()