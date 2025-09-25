import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as np
import math


iter = 100
T_memory = 10

start_switch = 30
dt = 60

goal_60 = list()
loss_60 = list()
data_time = list()
loss_memory_60 = list()
for i in range(iter):
    data = sio.loadmat('results/60/result_' + str(i) + '.mat')
    state = data['state'][0]  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']

    if i == 0:
        goal_60 = data['goal_error'][0]
        
        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_60 = loss_memory
    else:
        goal_60 = np.vstack((goal_60, data['goal_error'][0]))
        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_60 = np.hstack((loss_memory_60, loss_memory))

loss_60 = []
for i in range(iter):
    # loss_each = np.min(goal_60[start_switch, :start_switch+dt-1])
    loss_each = goal_60[i, start_switch+dt-2]
    loss_60.append(loss_each)


loss_avg_60 = np.mean(loss_60, 0)
loss_min_60 = np.min(loss_60, 0)
loss_max_60 = np.max(loss_60, 0)
loss_memory_avg_60 = np.mean(loss_memory_60, 0)
loss_memory_min_60 = np.min(loss_memory_60, 0)
loss_memory_max_60 = np.max(loss_memory_60, 0)


print("60 - Proposed:", loss_avg_60)

goal_60_OCIL = list()
loss_60_OCIL = list()
data_time_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/60_OCIL/result_' + str(i) + '.mat')
    state = data['state']  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']
    
    if i == 0:
        goal_60_OCIL = data['goal_error'][0]

        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_60_OCIL = loss_memory
    else:
        goal_60_OCIL = np.vstack((goal_60_OCIL, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_60_OCIL = np.hstack((loss_memory_60_OCIL, loss_memory))

loss_60_OCIL = []
for i in range(iter):
    # loss_each = np.min(goal_60_OCIL[i, :start_switch+dt-1])
    loss_each = goal_60_OCIL[i, start_switch+dt-2]
    loss_60_OCIL.append(loss_each)

loss_avg_60_OCIL = np.mean(loss_60_OCIL, 0)
loss_min_60_OCIL = np.min(loss_60_OCIL, 0)
loss_max_60_OCIL = np.max(loss_60_OCIL, 0)
loss_memory_avg_60_OCIL = np.mean(loss_memory_60_OCIL, 0)
loss_memory_min_60_OCIL = np.min(loss_memory_60_OCIL, 0)
loss_memory_max_60_OCIL = np.max(loss_memory_60_OCIL, 0)

print("60 - OCIL:", loss_avg_60_OCIL)


dt = 50
start_switch = 30
dt = 50

goal_50 = list()
loss_50 = list()
data_time_50 = list()
loss_memory_50 = list()
for i in range(iter):
    data = sio.loadmat('results/50/result_' + str(i) + '.mat')
    state = data['state'][0]  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']
    
    if i == 0:
        goal_50 = data['goal_error'][0]
        
        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_50 = loss_memory
    else:
        goal_50 = np.vstack((goal_50, data['goal_error'][0]))
        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_50 = np.hstack((loss_memory_50, loss_memory))

loss_50 = []
for i in range(iter):
    # loss_each = np.min(goal_50[start_switch, :start_switch+dt-1])
    loss_each = goal_50[i, start_switch+dt-2]
    loss_50.append(loss_each)

loss_avg_50 = np.mean(loss_50, 0)
loss_min_50 = np.min(loss_50, 0)
loss_max_50 = np.max(loss_50, 0)
loss_memory_avg_50 = np.mean(loss_memory_50, 0)
loss_memory_min_50 = np.min(loss_memory_50, 0)
loss_memory_max_50 = np.max(loss_memory_50, 0)

print("50 - Proposed:", loss_avg_50)

goal_50_OCIL = list()
loss_50_OCIL = list()
data_time_50_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/50_OCIL/result_' + str(i) + '.mat')
    state = data['state']  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']
    
    if i == 0:
        goal_50_OCIL = data['goal_error'][0]

        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_50_OCIL = loss_memory
    else:
        goal_50_OCIL = np.vstack((goal_50_OCIL, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_50_OCIL = np.hstack((loss_memory_50_OCIL, loss_memory))
    
loss_50_OCIL = []
for i in range(iter):
    # loss_each = np.min(goal_50_OCIL[i, :start_switch+dt-1])
    loss_each = goal_50_OCIL[i, start_switch+dt-2]
    loss_50_OCIL.append(loss_each)

loss_avg_50_OCIL = np.mean(loss_50_OCIL, 0)
loss_min_50_OCIL = np.min(loss_50_OCIL, 0)
loss_max_50_OCIL = np.max(loss_50_OCIL, 0)
loss_memory_avg_50_OCIL = np.mean(loss_memory_50_OCIL, 0)
loss_memory_min_50_OCIL = np.min(loss_memory_50_OCIL, 0)
loss_memory_max_50_OCIL = np.max(loss_memory_50_OCIL, 0)

print("50 - OCIL:", loss_avg_50_OCIL)


dt = 40
start_switch = 30

goal_40 = list()
loss_40 = list()
data_time_40 = list()
for i in range(iter):
    data = sio.loadmat('results/40/result_' + str(i) + '.mat')

    state = data['state'][0]  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']

    if i == 0:
        goal_40 = data['goal_error'][0]

        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_40 = loss_memory
    else:
        goal_40 = np.vstack((goal_40, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_40 = np.hstack((loss_memory_40, loss_memory))

loss_40 = []
for i in range(iter):
    # loss_each = np.min(goal_40[start_switch, :start_switch+dt-1])
    loss_each = goal_40[i, start_switch+dt-2]
    loss_40.append(loss_each)


loss_avg_40 = np.mean(loss_40, 0)
loss_min_40 = np.min(loss_40, 0)
loss_max_40 = np.max(loss_40, 0)
loss_memory_avg_40 = np.mean(loss_memory_40, 0)
loss_memory_min_40 = np.min(loss_memory_40, 0)
loss_memory_max_40 = np.max(loss_memory_40, 0)

print("40 - Proposed:", loss_avg_40)


goal_40_OCIL = list()
loss_40_OCIL = list()
data_time_40_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/40_OCIL/result_' + str(i) + '.mat')
    state = data['state']  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']

    if i == 0:
        goal_40_OCIL = data['goal_error'][0]

        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_40_OCIL = loss_memory
    else:
        goal_40_OCIL = np.vstack((goal_40_OCIL, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_40_OCIL = np.hstack((loss_memory_40_OCIL, loss_memory))

loss_40_OCIL = []
for i in range(iter):
    # loss_each = np.min(goal_40_OCIL[i, :start_switch+dt-1])
    loss_each = goal_40_OCIL[i, start_switch+dt-2]
    loss_40_OCIL.append(loss_each)

loss_avg_40_OCIL = np.mean(loss_40_OCIL, 0)
loss_min_40_OCIL = np.min(loss_40_OCIL, 0)
loss_max_40_OCIL = np.max(loss_40_OCIL, 0)
loss_memory_avg_40_OCIL = np.mean(loss_memory_40_OCIL, 0)
loss_memory_min_40_OCIL = np.min(loss_memory_40_OCIL, 0)
loss_memory_max_40_OCIL = np.max(loss_memory_40_OCIL, 0)

print("40 - OCIL:", loss_avg_40_OCIL)


dt = 30
start_switch = 50

goal_30 = list()
loss_30 = list()
data_time_30 = list()
for i in range(iter):
    data = sio.loadmat('results/30/result_' + str(i) + '.mat')
    state = data['state'][0]  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']

    if i == 0:
        goal_30 = data['goal_error'][0]

        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_30 = loss_memory
    else:
        goal_30 = np.vstack((goal_30, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_30 = np.hstack((loss_memory_30, loss_memory))

loss_30 = []
for i in range(iter):
    # loss_each = np.min(goal_30[start_switch, :start_switch+dt-1])
    loss_each = goal_30[i, start_switch+dt-2]
    loss_30.append(loss_each)

loss_avg_30 = np.mean(loss_30, 0)
loss_min_30 = np.min(loss_30, 0)
loss_max_30 = np.max(loss_30, 0)
loss_memory_avg_30 = np.mean(loss_memory_30, 0)
loss_memory_min_30 = np.min(loss_memory_30, 0)
loss_memory_max_30 = np.max(loss_memory_30, 0)

print("30 - Proposed:", loss_avg_30)

goal_30_OCIL = list()
loss_30_OCIL = list()
data_time_30_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/30_OCIL/result_' + str(i) + '.mat')
    state = data['state']  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']
    
    if i == 0:
        goal_30_OCIL = data['goal_error'][0]
        
        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_30_OCIL = loss_memory
    else:
        goal_30_OCIL = np.vstack((goal_30_OCIL, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_30_OCIL = np.hstack((loss_memory_30_OCIL, loss_memory))
        
loss_30_OCIL = []
for i in range(iter):
    # loss_each = np.min(goal_30_OCIL[i, :start_switch+dt-1])
    loss_each = goal_30_OCIL[i, start_switch+dt-2]
    loss_30_OCIL.append(loss_each)
    
loss_avg_30_OCIL = np.mean(loss_30_OCIL, 0)
loss_min_30_OCIL = np.min(loss_30_OCIL, 0)
loss_max_30_OCIL = np.max(loss_30_OCIL, 0)
loss_memory_avg_30_OCIL = np.mean(loss_memory_30_OCIL, 0)
loss_memory_min_30_OCIL = np.min(loss_memory_30_OCIL, 0)
loss_memory_max_30_OCIL = np.max(loss_memory_30_OCIL, 0)
print("30 - OCIL:", loss_avg_30_OCIL)


dt = 20
start_switch = 60

goal_20 = list()
loss_20 = list()
data_time_20 = list()
for i in range(iter):
    data = sio.loadmat('results/20/result_' + str(i) + '.mat')
    state = data['state'][0]  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']

    if i == 0:
        goal_20 = data['goal_error'][0]

        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_20 = loss_memory
    else:
        goal_20 = np.vstack((goal_20, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_20 = np.hstack((loss_memory_20, loss_memory))

loss_20 = []
for i in range(iter):
    # loss_each = np.min(goal_20[start_switch, :start_switch+dt-1])
    loss_each = goal_20[i, start_switch+dt-2]
    loss_20.append(loss_each)

loss_avg_20 = np.mean(loss_20, 0)
loss_min_20 = np.min(loss_20, 0)
loss_max_20 = np.max(loss_20, 0)
loss_memory_avg_20 = np.mean(loss_memory_20, 0)
loss_memory_min_20 = np.min(loss_memory_20, 0)
loss_memory_max_20 = np.max(loss_memory_20, 0)

print("20 - Proposed:", loss_avg_20)

loss_20_OCIL = list()
data_time_20_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/20_OCIL/result_' + str(i) + '.mat')
    state = data['state']  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']

    if i == 0:
        goal_20_OCIL = data['goal_error'][0]

        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_20_OCIL = loss_memory
    else:
        goal_20_OCIL = np.vstack((goal_20_OCIL, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_20_OCIL = np.hstack((loss_memory_20_OCIL, loss_memory))

loss_20_OCIL = []
for i in range(iter):
    # loss_each = np.min(goal_20_OCIL[i, :start_switch+dt-1])
    loss_each = goal_20_OCIL[i, start_switch+dt-2]
    loss_20_OCIL.append(loss_each)

loss_avg_20_OCIL = np.mean(loss_20_OCIL, 0)
loss_min_20_OCIL = np.min(loss_20_OCIL, 0)
loss_max_20_OCIL = np.max(loss_20_OCIL, 0)
loss_memory_avg_20_OCIL = np.mean(loss_memory_20_OCIL, 0)
loss_memory_min_20_OCIL = np.min(loss_memory_20_OCIL, 0)
loss_memory_max_20_OCIL = np.max(loss_memory_20_OCIL, 0)
print("20 - OCIL:", loss_avg_20_OCIL)

dt = 10
start_switch = 70

goal_10 = list()
loss_10 = list()
data_time_10 = list()
for i in range(iter):
    data = sio.loadmat('results/10/result_' + str(i) + '.mat')
    state = data['state'][0]  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']

    if i == 0:
        goal_10 = data['goal_error'][0]

        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_10 = loss_memory
    else:
        goal_10 = np.vstack((goal_10, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_10 = np.hstack((loss_memory_10, loss_memory))

loss_10 = []
for i in range(iter):
    # loss_each = np.min(goal_10[start_switch, :start_switch+dt-1])
    loss_each = goal_10[i, start_switch+dt-2]
    loss_10.append(loss_each)

loss_avg_10 = np.mean(loss_10, 0)
loss_min_10 = np.min(loss_10, 0)
loss_max_10 = np.max(loss_10, 0)
loss_memory_avg_10 = np.mean(loss_memory_10, 0)
loss_memory_min_10 = np.min(loss_memory_10, 0)
loss_memory_max_10 = np.max(loss_memory_10, 0)
print("10 - Proposed:", loss_avg_10)

loss_10_OCIL = list()
data_time_10_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/10_OCIL/result_' + str(i) + '.mat')
    state = data['state']  # Shape: (99, 13) - predicted trajectory
    theta = data['theta']
    demo_state = data['demo_state']

    if i == 0:
        goal_10_OCIL = data['goal_error'][0]

        loss_memory = []
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_10_OCIL = loss_memory
    else:
        goal_10_OCIL = np.vstack((goal_10_OCIL, data['goal_error'][0]))
        for t in range(len(state)):
            if t < T_memory:
                loss_memory.append(np.linalg.norm(state[t][:t]-demo_state[:t]))
            else:
                loss_memory.append(np.linalg.norm(state[t][:T_memory]-demo_state[t-T_memory:t]))
        loss_memory_10_OCIL = np.hstack((loss_memory_10_OCIL, loss_memory))

loss_10_OCIL = []
for i in range(iter):
    # loss_each = np.min(goal_10_OCIL[i, :start_switch+dt-1])
    loss_each = goal_10_OCIL[i, start_switch+dt-2]
    loss_10_OCIL.append(loss_each)

loss_avg_10_OCIL = np.mean(loss_10_OCIL, 0)
loss_min_10_OCIL = np.min(loss_10_OCIL, 0)
loss_max_10_OCIL = np.max(loss_10_OCIL, 0)
loss_memory_avg_10_OCIL = np.mean(loss_memory_10_OCIL, 0)
loss_memory_min_10_OCIL = np.min(loss_memory_10_OCIL, 0)
loss_memory_max_10_OCIL = np.max(loss_memory_10_OCIL, 0)
print("10 - OCIL:", loss_avg_10_OCIL)

# Calculate standard deviations for each case
std_20 = np.std(loss_20)
std_20_OCIL = np.std(loss_20_OCIL)
std_30 = np.std(loss_30)
std_30_OCIL = np.std(loss_30_OCIL)
std_40 = np.std(loss_40)
std_40_OCIL = np.std(loss_40_OCIL)
std_50 = np.std(loss_50)
std_50_OCIL = np.std(loss_50_OCIL)
std_60 = np.std(loss_60)
std_60_OCIL = np.std(loss_60_OCIL)

# Calculate standard deviations for memory loss
std_memory_20 = np.std(loss_memory_20)
std_memory_20_OCIL = np.std(loss_memory_20_OCIL)
std_memory_30 = np.std(loss_memory_30)
std_memory_30_OCIL = np.std(loss_memory_30_OCIL)
std_memory_40 = np.std(loss_memory_40)
std_memory_40_OCIL = np.std(loss_memory_40_OCIL)
std_memory_50 = np.std(loss_memory_50)
std_memory_50_OCIL = np.std(loss_memory_50_OCIL)
std_memory_60 = np.std(loss_memory_60)
std_memory_60_OCIL = np.std(loss_memory_60_OCIL)

# Create bar plot with error bars
fig, ax = plt.subplots(figsize=(6, 6))

# Data for plotting - cases ordered from 20 to 60 (excluding 10)
cases = [20, 30, 40, 50, 60]
means_data = [
    [loss_avg_20, loss_avg_20_OCIL], 
    [loss_avg_30, loss_avg_30_OCIL],
    [loss_avg_40, loss_avg_40_OCIL],
    [loss_avg_50, loss_avg_50_OCIL],
    [loss_avg_60, loss_avg_60_OCIL]
]
errors_data = [
    [std_20, std_20_OCIL],
    [std_30, std_30_OCIL],
    [std_40, std_40_OCIL],
    [std_50, std_50_OCIL],
    [std_60, std_60_OCIL]
]

colors = ['blue', 'green']  # Blue for Proposed, Green for OCIL
bar_width = 0.35

# Create bars for all cases
all_bars = []
all_means = []
all_errors = []

for i, case in enumerate(cases):
    x_pos = i
    means = means_data[i]
    errors = errors_data[i]
    
    bars = ax.bar([x_pos - bar_width/2, x_pos + bar_width/2], means, 
                  yerr=errors, capsize=8, alpha=0.7, width=bar_width,
                  color=colors, edgecolor='black', linewidth=1, 
                  error_kw={'capthick': 2, 'capsize': 8})
    
    all_bars.extend(bars)
    all_means.extend(means)
    all_errors.extend(errors)

# Customize the plot
ax.set_xlabel('Time step between switch', fontsize=20)
ax.set_ylabel('Average Prediction Loss', fontsize=20)
ax.set_xticks(range(len(cases)))
ax.set_xticklabels([str(case) for case in cases], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.grid(True, alpha=0.3, axis='y')

# Value labels removed for cleaner plot

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Proposed'),
                  Patch(facecolor='green', alpha=0.7, label='OCIL')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

ax.set_ylim(0, None)

plt.tight_layout()
plt.show()

# Create second plot for loss_memory
fig2, ax2 = plt.subplots(figsize=(6, 6))

# Data for plotting loss_memory - cases ordered from 20 to 60 (excluding 10)
cases_memory = [20, 30, 40, 50, 60]
means_memory_data = [
    [loss_memory_avg_20, loss_memory_avg_20_OCIL], 
    [loss_memory_avg_30, loss_memory_avg_30_OCIL],
    [loss_memory_avg_40, loss_memory_avg_40_OCIL],
    [loss_memory_avg_50, loss_memory_avg_50_OCIL],
    [loss_memory_avg_60, loss_memory_avg_60_OCIL]
]
errors_memory_data = [
    [std_memory_20, std_memory_20_OCIL],
    [std_memory_30, std_memory_30_OCIL],
    [std_memory_40, std_memory_40_OCIL],
    [std_memory_50, std_memory_50_OCIL],
    [std_memory_60, std_memory_60_OCIL]
]

colors_memory = ['blue', 'green']  # Blue for Proposed, Green for OCIL
bar_width_memory = 0.35

# Create bars for all cases
all_bars_memory = []
all_means_memory = []
all_errors_memory = []

for i, case in enumerate(cases_memory):
    x_pos = i
    means = means_memory_data[i]
    errors = errors_memory_data[i]
    
    bars = ax2.bar([x_pos - bar_width_memory/2, x_pos + bar_width_memory/2], means, 
                   yerr=errors, capsize=8, alpha=0.7, width=bar_width_memory,
                   color=colors_memory, edgecolor='black', linewidth=1, 
                   error_kw={'capthick': 2, 'capsize': 8})
    
    all_bars_memory.extend(bars)
    all_means_memory.extend(means)
    all_errors_memory.extend(errors)

# Customize the second plot
ax2.set_xlabel('Time step between switch', fontsize=20)
ax2.set_ylabel('Average Trajectory Loss', fontsize=20)
ax2.set_xticks(range(len(cases_memory)))
ax2.set_xticklabels([str(case) for case in cases_memory], fontsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax2.grid(True, alpha=0.3, axis='y')

# Add legend for second plot
legend_elements_memory = [Patch(facecolor='blue', alpha=0.7, label='Proposed'),
                         Patch(facecolor='green', alpha=0.7, label='OCIL')]
# ax2.legend(handles=legend_elements_memory, loc='upper right', fontsize=20)

ax2.set_ylim(1e-1, None)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()




