import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

# Load the .mat files
dt_1_data = sio.loadmat('results/dt_1.mat')
dt_2_data = sio.loadmat('results/dt_2.mat')
dt_5_data = sio.loadmat('results/dt_5.mat')
dt_10_data = sio.loadmat('results/dt_10.mat')

# Extract goal error data
goal_error_dt_1 = dt_1_data['goal_error'][0]
goal_error_dt_2 = dt_2_data['goal_error'][0]
goal_error_dt_5 = dt_5_data['goal_error'][0]
goal_error_dt_10 = dt_10_data['goal_error'][0]

# Create time array (assuming time steps are sequential)
time_steps = np.arange(len(goal_error_dt_1))

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot goal error vs time for each dt
ax.plot(time_steps, goal_error_dt_1, label='$\Delta$ = 1', linewidth=2, color='blue')
ax.plot(time_steps, goal_error_dt_2, label='$\Delta$ = 2', linewidth=4, color='red')
ax.plot(time_steps, goal_error_dt_5, label='$\Delta$ = 5', linewidth=4, color='green')
ax.plot(time_steps, goal_error_dt_10, label='$\Delta$ = 10', linewidth=4, color='orange')
ax.plot(time_steps, goal_error_dt_5, linewidth=4, color='green')
ax.plot(time_steps, goal_error_dt_2, linewidth=4, color='red')
ax.plot(time_steps, goal_error_dt_1, linewidth=2, color='blue')

# Customize the plot
ax.set_xlabel('$t$', fontsize=20)
ax.set_ylabel('Prediction Loss', fontsize=20)
ax.legend(fontsize=20)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)

# Set axis properties
ax.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
plt.show()
