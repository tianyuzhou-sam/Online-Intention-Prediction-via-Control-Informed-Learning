import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np

# Load switch_target.mat data
data = sio.loadmat('results/switch_target.mat')

# Extract data
goal_error = data['goal_error'][0]  # Shape: (99,)
data_time = data['data_time'][0]    # Shape: (99,)
demo_state_noise = data['demo_state_noise']  # Shape: (101, 13)
state = data['state'][0]  # Shape: (99, 13) - predicted trajectory
theta = data['theta']

# Extract position coordinates (x, y, z) from demo_state_noise
# State structure: [rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
x_pos = demo_state_noise[:, 0]  # rx
y_pos = demo_state_noise[:, 1]  # ry  
z_pos = demo_state_noise[:, 2]  # rz

# Create timestep array
timestep = list(range(0, len(goal_error)))

# Set up plotting parameters
plt.rcParams['font.size'] = 20

# First plot: Prediction error
fig1, ax1 = plt.subplots(figsize=(10, 6))
line_goal, = ax1.plot(goal_error, color='b', linewidth=4)
ax1.set_xlabel('$t$')
ax1.set_ylabel('Prediction Loss')
ax1.set_title('Prediction Error for Switch Target')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, max(goal_error) * 1.1])
plt.tight_layout()
plt.show()




goal_position = [2,10,1]
switch_time = [20,60]
switch_goal = [[10,5,1],[10,10,1]]

# Second plot: 2D trajectory animation (top view only)
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
# ax2.set_title('2D Trajectory Animation (Top View)')
ax2.grid(True, alpha=0.3)

# Set axis limits based on trajectory bounds (considering both actual and predicted)
margin = 0.2
x_min = min(x_pos)
x_max = max(x_pos)
y_min = min(y_pos)
y_max = max(y_pos)
x_range = x_max - x_min
y_range = y_max - y_min
ax2.set_xlim([x_min - margin * x_range, x_max + margin * x_range])
ax2.set_ylim([y_min - margin * y_range, y_max + margin * y_range])

# Initialize animation elements
trajectory_line, = ax2.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Actual Trajectory')
predicted_line, = ax2.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Predicted Trajectory')
current_point, = ax2.plot([], [], 'ro', markersize=8, label='Current State')
goal_point, = ax2.plot([], [], 'r*', markersize=15, label='Goal Position')
predicted_goal_point, = ax2.plot([], [], 'b*', markersize=12, label='Predicted Goal')

# Add legend at right-bottom
ax2.legend(loc='lower right')

# Animation function
def animate(frame):
    # Update trajectory lines (show full trajectory up to current frame)
    trajectory_line.set_data(x_pos[:frame+1], y_pos[:frame+1])
    
    # Update predicted trajectory (show full predicted trajectory up to current frame)
    if frame < len(state):
        predicted_traj = state[frame]
        x_pred_frame = predicted_traj[:, 0]  # x positions for this frame
        y_pred_frame = predicted_traj[:, 1]  # y positions for this frame
        predicted_line.set_data(x_pred_frame, y_pred_frame)
        
        # Extract predicted goal from theta history (elements 11-12 of the state)
        # theta contains the goal position in elements 11-12
        predicted_goal_x = theta[frame][11]  # x coordinate of predicted goal
        predicted_goal_y = theta[frame][12]  # y coordinate of predicted goal
        predicted_goal_point.set_data([predicted_goal_x], [predicted_goal_y])
    
    # Update current position point
    if frame < len(x_pos):
        current_point.set_data([x_pos[frame]], [y_pos[frame]])
    
    # Determine current goal based on switch times
    current_goal = goal_position
    for i, switch_t in enumerate(switch_time):
        if frame >= switch_t:
            current_goal = switch_goal[i]
    
    # Update goal position
    goal_point.set_data([current_goal[0]], [current_goal[1]])
    
    return trajectory_line, predicted_line, current_point, goal_point, predicted_goal_point

# Create animation
anim = animation.FuncAnimation(fig2, animate, frames=len(x_pos), 
                             interval=150, blit=True, repeat=True)

plt.tight_layout()
plt.show()

# Save the plot
# plt.savefig('results/switch_target_prediction_error.png', dpi=300, bbox_inches='tight')
# print("Plot saved as 'results/switch_target_prediction_error.png'")


