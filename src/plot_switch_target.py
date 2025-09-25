import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import scipy.io as sio
import numpy as np

# Load switch_target.mat data
data = sio.loadmat('results/switch_target.mat')
T_memory = 10

# Extract data
goal_error = data['goal_error'][0]  # Shape: (99,)
data_time = data['data_time'][0]    # Shape: (99,)
demo_state_noise = data['demo_state_noise']  # Shape: (101, 13)
state = data['state'][0]  # Shape: (99, 13) - predicted trajectory
theta = data['theta']
demo_state = data['demo_state']


loss_memory = []
for i in range(len(state)):
    if i < T_memory:
        loss_memory.append(np.linalg.norm(state[i][:i]-demo_state[:i]))
    else:
        loss_memory.append(np.linalg.norm(state[i][:T_memory]-demo_state[i-T_memory:i]))

# Extract position coordinates (x, y, z) from demo_state_noise
# State structure: [rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
x_pos = demo_state_noise[:, 0]  # rx
y_pos = demo_state_noise[:, 1]  # ry  
z_pos = demo_state_noise[:, 2]  # rz

# Create timestep array
timestep = list(range(0, len(goal_error)))

OCIL_data = sio.loadmat('results/OCIL_prediction.mat')
OCIL_goal_error = OCIL_data['goal_error'][0]
OCIL_data_time = OCIL_data['data_time'][0]
OCIL_state = OCIL_data['state']
OCIL_theta = OCIL_data['theta']

loss_memory_OCIL = []
for i in range(len(state)):
    if i < T_memory:
        loss_memory_OCIL.append(np.linalg.norm(OCIL_state[i][:i]-demo_state[:i]))
    else:
        loss_memory_OCIL.append(np.linalg.norm(OCIL_state[i][:T_memory]-demo_state[i-T_memory:i]))


plot_OCIL = True

# Set up plotting parameters
plt.rcParams['font.size'] = 24

# First plot: Prediction error
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.axvline(x=50, color='k', linestyle='--', linewidth=2, alpha=0.5)
ax1.axvline(x=80, color='k', linestyle='--', linewidth=2, alpha=0.5)
line_goal, = ax1.plot(goal_error, color='b', linewidth=5)
if plot_OCIL:
    line_OCIL_goal, = ax1.plot(OCIL_goal_error, color='g', linestyle='--', linewidth=3)
ax1.set_xlabel('$t$')
ax1.set_ylabel('Prediction Loss')
ax1.legend([line_goal, line_OCIL_goal], ['Proposed', 'OCIL'], loc='upper left', fontsize=18)
# ax1.set_title('Prediction Error for Switch Target')
# ax1.grid(True, alpha=0.3)

ax1.set_xlim(0, 120)
ax1.set_ylim([0, max(OCIL_goal_error) * 1.1])
plt.tight_layout()
plt.show()




goal_position = [2,10,1]
switch_time = [50,80]
switch_goal = [[10,8,1],[0,5,1]]

# H = 120
# goal_position = [5,10,1]
# switch_time = list(range(60, H, 1))
# switch_goal = []
# for idx in range(H):
#     if idx >= 60:
#         if idx < 90:
#             goal_position[0] = goal_position[0] + 0.05
#         else:
#             goal_position[0] = goal_position[0] - 0.05
#         goal_position[1] = goal_position[1] - 0.05
#         # Create a new list copy to avoid reference issues
#         switch_goal.append([goal_position[0], goal_position[1], goal_position[2]])



# Second plot: 2D trajectory animation (top view only)
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
# ax2.set_title('2D Trajectory Animation (Top View)')
# ax2.grid(True, alpha=0.3)

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
ax2.set_xlim([-2, 12])
ax2.set_ylim([-2, 12])

# Initialize animation elements
trajectory_line, = ax2.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Trajectory')
predicted_line, = ax2.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Prediction')
current_point, = ax2.plot([], [], 'ro', markersize=8, label='Current State')
goal_point, = ax2.plot([], [], 'r*', markersize=15, label='Goal')
predicted_goal_point, = ax2.plot([], [], 'b*', markersize=12, label='Predicted Goal')

# Add time text in top left corner
time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=16, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
    
    # Update time text (show time step from 0 to 120)
    time_text.set_text(f'$t = {frame}$')
    
    return trajectory_line, predicted_line, current_point, goal_point, predicted_goal_point, time_text

# Create animation
anim = animation.FuncAnimation(fig2, animate, frames=len(x_pos), 
                             interval=150, blit=True, repeat=True)

# Save animation
print("Saving animation...")
anim.save('results/trajectory_animation.mp4', writer='ffmpeg', fps=10, dpi=300)
print("Animation saved as 'results/trajectory_animation.mp4'")

plt.tight_layout()
plt.show()

# Third plot: Three subplots showing snapshots at different time steps
frame_list = [49, 75, 95]  # List of time steps to display
fig3, ax3 = plt.subplots(1,3, figsize=(15, 5))

# Subplot 1: Time step from frame_list[0]
frame_0 = frame_list[0]
ax3[0].set_xlabel('$x$')
ax3[0].set_ylabel('$y$')
# ax3[0].grid(True, alpha=0.3)
ax3[0].set_xlim([-2, 12])
ax3[0].set_ylim([-2, 12])

# Plot full trajectory up to time step
if frame_0 < len(state):
    if plot_OCIL:
        predicted_traj_0_OCIL = OCIL_state[frame_0]
        x_pred_0_OCIL = predicted_traj_0_OCIL[:, 0]
        y_pred_0_OCIL = predicted_traj_0_OCIL[:, 1]
        ax3[0].plot(x_pred_0_OCIL, y_pred_0_OCIL, 'g--', linewidth=3, alpha=0.7, label='Prediction (OCIL)')

ax3[0].plot(x_pos[:frame_0+1], y_pos[:frame_0+1], 'r-', linewidth=5, alpha=0.7, label='Trajectory')

# Plot predicted trajectory at time step
if frame_0 < len(state):
    predicted_traj_0 = state[frame_0]
    x_pred_0 = predicted_traj_0[:, 0]
    y_pred_0 = predicted_traj_0[:, 1]
    ax3[0].plot(x_pred_0, y_pred_0, 'b--', linewidth=3, alpha=0.7, label='Prediction (Proposed)')

# Plot current position at time step
ax3[0].plot(x_pos[frame_0], y_pos[frame_0], 'ro', markersize=15, label='Current State')

# Determine goal at time step
current_goal_0 = goal_position
for i, switch_t in enumerate(switch_time):
    if frame_0 >= switch_t:
        current_goal_0 = switch_goal[i]

# Plot goal position at time step
ax3[0].plot(current_goal_0[0], current_goal_0[1], 'r^', markersize=15, label='Goal')

# Plot predicted goal at time step
if frame_0 < len(theta):
    predicted_goal_x_0 = theta[frame_0][11]
    predicted_goal_y_0 = theta[frame_0][12]
    ax3[0].plot(predicted_goal_x_0, predicted_goal_y_0, 'b*', markersize=15, label='Predicted Goal (Proposed)')
    if plot_OCIL:
        predicted_goal_x_0_OCIL = OCIL_theta[frame_0][11]
        predicted_goal_y_0_OCIL = OCIL_theta[frame_0][12]
        ax3[0].plot(predicted_goal_x_0_OCIL, predicted_goal_y_0_OCIL, 'g*', markersize=15, label='Predicted Goal (OCIL)')

# Add legend to first subplot
# ax3[0].legend(loc='lower right', bbox_to_anchor=(1.3, 0.02))

# Add time step text
# ax3[0].text(0.02, 0.98, 'Time: 39', transform=ax3[0].transAxes, fontsize=16, 
        #    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Create custom legend with specified order
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import matplotlib.lines as mlines

import matplotlib.lines as mlines

# row 1 handles
row1 = [
    mlines.Line2D([0], [0], color='red', linewidth=5, label='Trajectory'),
    mlines.Line2D([0], [0], color='blue', linestyle='--', linewidth=3, label='Prediction (Proposed)'),
    mlines.Line2D([0], [0], color='green', linestyle='--', linewidth=3, label='Prediction (OCIL)'),
    mlines.Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=15, label='Current State')
]

# row 2 handles
row2 = [
    mlines.Line2D([0], [0], color='red', marker='^', linestyle='None', markersize=15, label='Goal'),
    mlines.Line2D([0], [0], color='blue', marker='*', linestyle='None', markersize=15, label='Predicted Goal (Proposed)'),
    mlines.Line2D([0], [0], color='green', marker='*', linestyle='None', markersize=15, label='Predicted Goal (OCIL)')
]

leg1 = fig3.legend(handles=row1,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.03),
                   ncol=4,
                   fontsize=20)

leg2 = fig3.legend(handles=row2,
                   loc='upper center',
                   bbox_to_anchor=(0.59, 0.95),
                   ncol=3,
                   fontsize=20)

fig3.add_artist(leg1)  # so the first legend stays




            
# Subplot 2: Time step from frame_list[1]
frame_1 = frame_list[1]
ax3[1].set_xlabel('$x$')
# ax3[1].grid(True, alpha=0.3)
ax3[1].set_xlim([-2, 12])
ax3[1].set_ylim([-2, 12])

# Plot full trajectory up to time step
if frame_1 < len(state):
    if plot_OCIL:
        predicted_traj_1_OCIL = OCIL_state[frame_1]
        x_pred_1_OCIL = predicted_traj_1_OCIL[:, 0]
        y_pred_1_OCIL = predicted_traj_1_OCIL[:, 1]
        ax3[1].plot(x_pred_1_OCIL, y_pred_1_OCIL, 'g--', linewidth=3, alpha=0.7, label='Prediction (OCIL)')
ax3[1].plot(x_pos[:frame_1+1], y_pos[:frame_1+1], 'r-', linewidth=5, alpha=0.7, label='Trajectory')

# Plot predicted trajectory at time step
if frame_1 < len(state):
    predicted_traj_1 = state[frame_1]
    x_pred_1 = predicted_traj_1[:, 0]
    y_pred_1 = predicted_traj_1[:, 1]
    ax3[1].plot(x_pred_1, y_pred_1, 'b--', linewidth=3, alpha=0.7, label='Prediction (Proposed)')
# Plot current position at time step
ax3[1].plot(x_pos[frame_1], y_pos[frame_1], 'ro', markersize=15, label='Current State')

# Determine goal at time step
current_goal_1 = goal_position
for i, switch_t in enumerate(switch_time):
    if frame_1 >= switch_t:
        current_goal_1 = switch_goal[i]

# Plot goal position at time step
ax3[1].plot(current_goal_1[0], current_goal_1[1], 'r^', markersize=15, label='Goal')

# Plot predicted goal at time step
if frame_1 < len(theta):
    predicted_goal_x_1 = theta[frame_1][11]
    predicted_goal_y_1 = theta[frame_1][12]
    ax3[1].plot(predicted_goal_x_1, predicted_goal_y_1, 'b*', markersize=15, label='Predicted Goal (Proposed)')
    if plot_OCIL:
        predicted_goal_x_1_OCIL = OCIL_theta[frame_1][11]
        predicted_goal_y_1_OCIL = OCIL_theta[frame_1][12]
        ax3[1].plot(predicted_goal_x_1_OCIL, predicted_goal_y_1_OCIL, 'g*', markersize=15, label='Predicted Goal (OCIL)')

# Add time step text
# ax3[1].text(0.02, 0.98, 'Time: 75', transform=ax3[1].transAxes, fontsize=16, 
        #    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Subplot 3: Time step from frame_list[2]
frame_2 = frame_list[2]
ax3[2].set_xlabel('$x$')
# ax3[2].grid(True, alpha=0.3)
ax3[2].set_xlim([-2, 12])
ax3[2].set_ylim([-2, 12])

# Plot full trajectory up to time step
if frame_2 < len(state):
    if plot_OCIL:
        predicted_traj_2_OCIL = OCIL_state[frame_2]
        x_pred_2_OCIL = predicted_traj_2_OCIL[:, 0]
        y_pred_2_OCIL = predicted_traj_2_OCIL[:, 1]
        ax3[2].plot(x_pred_2_OCIL, y_pred_2_OCIL, 'g--', linewidth=3, alpha=0.7, label='Prediction (OCIL)')
ax3[2].plot(x_pos[:frame_2+1], y_pos[:frame_2+1], 'r-', linewidth=5, alpha=0.7, label='Trajectory')

# Plot predicted trajectory at time step
if frame_2 < len(state):
    predicted_traj_2 = state[frame_2]
    x_pred_2 = predicted_traj_2[:, 0]
    y_pred_2 = predicted_traj_2[:, 1]
    ax3[2].plot(x_pred_2, y_pred_2, 'b--', linewidth=3, alpha=0.7, label='Prediction (Proposed)')
# Plot current position at time step
ax3[2].plot(x_pos[frame_2], y_pos[frame_2], 'ro', markersize=15, label='Current State')

# Determine goal at time step
current_goal_2 = goal_position
for i, switch_t in enumerate(switch_time):
    if frame_2 >= switch_t:
        current_goal_2 = switch_goal[i]

# Plot goal position at time step
ax3[2].plot(current_goal_2[0], current_goal_2[1], 'r^', markersize=15, label='Goal')

# Plot predicted goal at time step
if frame_2 < len(theta):
    predicted_goal_x_2 = theta[frame_2][11]
    predicted_goal_y_2 = theta[frame_2][12]
    ax3[2].plot(predicted_goal_x_2, predicted_goal_y_2, 'b*', markersize=15, label='Predicted Goal (Proposed)')
    if plot_OCIL:
        predicted_goal_x_2_OCIL = OCIL_theta[frame_2][11]
        predicted_goal_y_2_OCIL = OCIL_theta[frame_2][12]
        ax3[2].plot(predicted_goal_x_2_OCIL, predicted_goal_y_2_OCIL, 'g*', markersize=15, label='Predicted Goal (OCIL)')
# Add time step text
# ax3[2].text(0.02, 0.98, 'Time: 90', transform=ax3[2].transAxes, fontsize=16, 
#            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Save the plot
# plt.savefig('results/switch_target_prediction_error.png', dpi=300, bbox_inches='tight')
# print("Plot saved as 'results/switch_target_prediction_error.png'")

fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.axvline(x=50, color='k', linestyle='--', linewidth=2, alpha=0.5)
ax3.axvline(x=80, color='k', linestyle='--', linewidth=2, alpha=0.5)
line_loss_memory, = ax3.plot(loss_memory, color='b', linewidth=5, label='Proposed')
if plot_OCIL:
    line_OCIL_loss_memory, = ax3.plot(loss_memory_OCIL, color='g', linestyle='--', linewidth=3)
ax3.set_xlabel('$t$')
ax3.set_ylabel('Trajectory Loss')
ax3.legend(loc='upper left', fontsize=18)
ax3.legend([line_loss_memory, line_OCIL_loss_memory], ['Proposed', 'OCIL'], loc='upper left', fontsize=18)
# ax3.set_title('Prediction Error for Switch Target')
# ax1.grid(True, alpha=0.3)

ax3.set_xlim(0, 120)
# ax3.set_ylim([0, max(loss_memory) * 1.1])
plt.tight_layout()
plt.show()




###########################################################
# Load switch_target.mat data
data2 = sio.loadmat('results/switch_target2.mat')

# Extract data
goal_error2 = data2['goal_error'][0]  # Shape: (99,)
state2 = data2['state'][0]
demo_state2 = data2['demo_state']
theta2 = data2['theta']

data2_OCIL = sio.loadmat('results/OCIL_prediction2.mat')
goal_error2_OCIL = data2_OCIL['goal_error'][0]
state2_OCIL = data2_OCIL['state']
theta2_OCIL = data2_OCIL['theta']

goal = [-4,4,1]
switch_time = [50,80,100]
pred_time = [49,79,95,115]
switch_goal = [[3,2,1],[4,5,1],[0,5,1]]

print(demo_state2)

fig4, ax4 = plt.subplots(figsize=(8, 8))
ax4.plot(demo_state2[:,0], demo_state2[:,1], 'r', linewidth=10, label='Trajectory')
ax4.plot(goal[0], goal[1], 'r^', markersize=20, label='Goal')

# Plot switch goals and goals without labels to avoid duplicate legend entries
for i in range(len(switch_time)):
    ax4.plot(demo_state2[switch_time[i],0], demo_state2[switch_time[i],1], 'yo', markersize=20)
    ax4.plot(switch_goal[i][0], switch_goal[i][1], 'r^', markersize=20)

# Plot predictions and predicted goals without labels to avoid duplicate legend entries
for i in range(len(pred_time)):
    ax4.plot(state2[pred_time[i]][:,0], state2[pred_time[i]][:,1], 'b--', linewidth=5, alpha=0.7)
    ax4.plot(theta2[pred_time[i]][11], theta2[pred_time[i]][12], 'b*', markersize=20)
    # ax4.plot(state2_OCIL[pred_time[i]][:,0], state2_OCIL[pred_time[i]][:,1], 'g--', linewidth=3, alpha=0.7, label='Prediction (OCIL)')
    # ax4.plot(theta2_OCIL[pred_time[i]][11], theta2_OCIL[pred_time[i]][12], 'g*', markersize=15, label='Predicted Goal (OCIL)')

# Add single legend entries for each type
# ax4.plot([], [], 'yo', markersize=20, label='Switch Goal Location')
# ax4.plot([], [], 'b--', linewidth=6, alpha=0.7, label='Prediction')
# ax4.plot([], [], 'b*', markersize=20, label='Predicted Goal')

# ax4.legend(loc='lower right', fontsize=20)
ax4.set_ylabel('$y$', fontsize=30)
ax4.set_xlabel('$x$', fontsize=30)
ax4.tick_params(axis='both', labelsize=30)
ax4.set_xlim([-4.5, 6])
ax4.set_ylim([-0.2, 5.5])

plt.tight_layout()
plt.show()


fig4, ax4 = plt.subplots(figsize=(8, 8))
ax4.plot(demo_state2[:,0], demo_state2[:,1], 'r', linewidth=5)
ax4.plot(goal[0], goal[1], 'r^', markersize=20)

# Plot switch goals and goals without labels to avoid duplicate legend entries
for i in range(len(switch_time)):
    ax4.plot(demo_state2[switch_time[i],0], demo_state2[switch_time[i],1], 'yo', markersize=20)
    ax4.plot(switch_goal[i][0], switch_goal[i][1], 'r^', markersize=20)

# Plot predictions and predicted goals without labels to avoid duplicate legend entries
for i in range(len(pred_time)):
    # ax4.plot(state2[pred_time[i]][:,0], state2[pred_time[i]][:,1], 'b--', linewidth=3, alpha=0.7)
    # ax4.plot(theta2[pred_time[i]][11], theta2[pred_time[i]][12], 'b*', markersize=15)
    ax4.plot(state2_OCIL[pred_time[i]][:,0], state2_OCIL[pred_time[i]][:,1], 'g--', linewidth=5, alpha=0.7)
    ax4.plot(theta2_OCIL[pred_time[i]][11], theta2_OCIL[pred_time[i]][12], 'g*', markersize=20)

# Add single legend entries for each type
# ax4.plot([], [], 'go', markersize=15, label='Switch Goal Location')
# ax4.plot([], [], 'g--', linewidth=3, alpha=0.7, label='Prediction (OCIL)')
# ax4.plot([], [], 'g*', markersize=15, label='Predicted Goal (OCIL)')

# ax4.legend(loc='lower right', fontsize=18)
ax4.set_ylabel('$y$', fontsize=30)
ax4.set_xlabel('$x$', fontsize=30)
ax4.tick_params(axis='both', labelsize=30)
ax4.set_xlim([-4.5, 6])
ax4.set_ylim([-0.2, 5.5])

plt.tight_layout()


fig4, ax4 = plt.subplots(figsize=(8, 8))
ax4.plot([], [], 'r', linewidth=5, label='Trajectory')
ax4.plot([], [], 'r^', markersize=20, label='Goal')
ax4.plot([], [], 'yo', markersize=20, label='Switch Goal Location')
ax4.plot([], [], 'b--', linewidth=5, alpha=0.7, label='Prediction (Proposed)')
ax4.plot([], [], 'b*', markersize=20, label='Predicted Goal (Proposed)')
ax4.plot([], [], 'g--', linewidth=5, alpha=0.7, label='Prediction (OCIL)')
ax4.plot([], [], 'g*', markersize=20, label='Predicted Goal (OCIL)')

ax4.legend(loc='lower right', fontsize=20)

plt.show()

