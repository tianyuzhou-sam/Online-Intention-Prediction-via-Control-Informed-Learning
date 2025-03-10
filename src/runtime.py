import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as np
import math


iter = 1000
goal = list()
loss = list()
data_time = list()
for i in range(iter):
    data = sio.loadmat('results/time/partial/result_' + str(i) + '.mat')
    
    if i == 0:
        goal = data['goal_error'][0]
        loss = data['Loss'][0]
        data_time = data['data_time'][0].tolist()
        grad_time = data['gradient_time'][0].tolist()
        ekf_time = data['ekf_time'][0].tolist()
    else:
        goal = np.vstack((goal, data['goal_error'][0]))
        loss = np.vstack((loss, data['Loss'][0]))
        data_time = np.hstack((data_time, data['data_time'][0].tolist()))
        grad_time = np.hstack((grad_time, data['gradient_time'][0].tolist()))
        ekf_time = np.hstack((ekf_time, data['ekf_time'][0].tolist()))


timestep = list(range(0, len(goal[0])))

time_avg = np.mean(data_time, 0)
time_std = np.std(data_time, 0)
grad_avg = np.mean(grad_time, 0)
grad_std = np.std(grad_time, 0)
ekf_avg = np.mean(ekf_time, 0)
ekf_std = np.std(ekf_time, 0)
print('Average time: ', time_avg)
print('Std time: ', time_std)
print('Average Grad: ', grad_avg)
print('Std Grad: ', grad_std)
print('Average EKF: ', ekf_avg)
print('Std EKF: ', ekf_std)

iter = 10
goal = list()
loss = list()
data_time = list()
for i in range(iter):
    data = sio.loadmat('results/time/nn/nn_result_' + str(i) + '.mat')
    
    if i == 0:
        goal = data['goal_error'][0]
        loss = data['Loss'][0]
        data_time = data['data_time'][0].tolist()
        grad_time = data['gradient_time'][0].tolist()
        ekf_time = data['ekf_time'][0].tolist()
    else:
        goal = np.vstack((goal, data['goal_error'][0]))
        loss = np.vstack((loss, data['Loss'][0]))
        data_time = np.hstack((data_time, data['data_time'][0].tolist()))
        grad_time = np.hstack((grad_time, data['gradient_time'][0].tolist()))
        ekf_time = np.hstack((ekf_time, data['ekf_time'][0].tolist()))


timestep = list(range(0, len(goal[0])))

time_avg = np.mean(data_time, 0)
time_std = np.std(data_time, 0)
grad_avg = np.mean(grad_time, 0)
grad_std = np.std(grad_time, 0)
ekf_avg = np.mean(ekf_time, 0)
ekf_std = np.std(ekf_time, 0)
print('run time for nn')
print('Average time: ', time_avg)
print('Std time: ', time_std)
print('Average Grad: ', grad_avg)
print('Std Grad: ', grad_std)
print('Average EKF: ', ekf_avg)
print('Std EKF: ', ekf_std)