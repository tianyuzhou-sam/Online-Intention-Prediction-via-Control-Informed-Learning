import numpy as np
import scipy.io as sio
import csv

# Load the .mat file
data = sio.loadmat('experiment/quad_demos.mat')

# Get the trajectory data
trajectory = data['trajectories']['state_traj_opt'][0][0]

# Create time array (assuming dt = 0.1)
dt = 0.1
time = np.arange(0, trajectory.shape[0] * dt, dt)

# Extract position and velocity data
x = trajectory[:, 0]  # x position
y = trajectory[:, 1]  # y position
z = trajectory[:, 2]  # z position
vx = trajectory[:, 3]  # x velocity
vy = trajectory[:, 4]  # y velocity
vz = trajectory[:, 5]  # z velocity

# Write to CSV file
with open('experiment/quad.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write time in first row
    writer.writerow(time)
    
    # Write position data (x, y, z)
    writer.writerow(x)
    writer.writerow(y)
    writer.writerow(z)
    
    # Write velocity data (vx, vy, vz)
    writer.writerow(vx)
    writer.writerow(vy)
    writer.writerow(vz)

print("Data has been written to quad.csv")
