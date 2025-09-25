import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

# Load the .mat files
Tm_5_data = sio.loadmat('results/Tm_5.mat')
Tm_10_data = sio.loadmat('results/Tm_10.mat')
Tm_20_data = sio.loadmat('results/Tm_20.mat')

# Extract goal error data
goal_error_Tm_5 = Tm_5_data['goal_error'][0]
goal_error_Tm_10 = Tm_10_data['goal_error'][0]
goal_error_Tm_20 = Tm_20_data['goal_error'][0]

# Create time array (assuming time steps are sequential)
time_steps = np.arange(len(goal_error_Tm_5))

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot goal error vs time for each Tm
ax.plot(time_steps, goal_error_Tm_5, label='$T_m$ = 5', linewidth=4, color='red')
ax.plot(time_steps, goal_error_Tm_10, label='$T_m$ = 10', linewidth=4, color='blue')
ax.plot(time_steps, goal_error_Tm_20, label='$T_m$ = 20', linewidth=4, color='green')

# Customize the plot
ax.set_xlabel('$t$', fontsize=20)
ax.set_ylabel('Prediction Loss', fontsize=20)
ax.legend(fontsize=20)
ax.grid(True, alpha=0.3)

# Set axis properties
ax.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
plt.show()
