import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import csv
import os
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.ticker as ticker

# Set global font sizes
plt.rcParams.update({
    'font.size': 18,  # Base font size
    'axes.titlesize': 24,  # Title font size
    'axes.labelsize': 20,  # Axis label font size
    'xtick.labelsize': 18,  # X-axis tick label font size
    'ytick.labelsize': 18,  # Y-axis tick label font size
    'legend.fontsize': 18,  # Legend font size
})

def load_dog_data():
    """Load and process the dog data from CSV file"""
    with open('experiment/dog_data.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in csv_reader:
            data.append([float(val) for val in row])
        demo_traj = np.array(data)
    
    # Normalize timestamps
    demo_traj[:,0] = demo_traj[:,0] - demo_traj[0, 0]
    
    # Filter data for specific time intervals
    start_time = 9
    dt = 0.1
    horizon = 100
    mask = (demo_traj[:,0] >= start_time) & (demo_traj[:,0] <= start_time + dt*horizon)
    filtered_indices = np.where(mask)[0]
    
    # Get samples at dt intervals
    time_points = np.arange(start_time, start_time + dt*horizon + dt, dt)
    sampled_indices = []
    for t in time_points:
        idx = np.abs(demo_traj[filtered_indices, 0] - t).argmin()
        sampled_indices.append(filtered_indices[idx])
    
    demo_traj = demo_traj[sampled_indices]
    demo_traj = demo_traj[:(horizon+1),:]
    
    # Extract state trajectory (x, y, yaw)
    demo_state_traj = np.column_stack((demo_traj[:,1], demo_traj[:,2], demo_traj[:,6]))
    
    return demo_state_traj, demo_traj

def load_prediction_results():
    """Load the prediction results from .mat file"""
    results_path = "experiment/predictiondog.mat"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    return sio.loadmat(results_path)

def plot_trajectories(demo_state_traj, results):
    """Plot the state trajectories"""
    state = results['state'][-1]  # Get the final state trajectory
    
    # Create time array
    t = np.arange(len(demo_state_traj))
    
    # Plot position trajectories
    fig, axs = plt.subplots(2, 1, figsize=(10,8))
    
    # Plot x position
    axs[0].plot(t, demo_state_traj[:, 0], 'r-', label='Real Trajectory')
    axs[0].plot(t, state[:, 0], 'b', label='Learned')
    axs[0].set_ylabel('x position', fontsize=20)
    axs[0].tick_params(axis='both', which='major', labelsize=18)
    axs[0].legend(fontsize=18)
    axs[0].set_ylim(-2.5, 2.5)  # Set y-axis limits for x position
    
    # Plot y position
    axs[1].plot(t, demo_state_traj[:, 1], 'r-', label='Real Trajectory')
    axs[1].plot(t, state[:, 1], 'b', label='Learned')
    axs[1].set_ylabel('y position', fontsize=20)
    axs[1].set_xlabel('Time steps', fontsize=20)
    axs[1].tick_params(axis='both', which='major', labelsize=18)
    axs[1].legend(fontsize=18)
    axs[1].set_ylim(-2.5, 2.5)  # Set y-axis limits for y position
    
    plt.suptitle('Position Trajectories', fontsize=24)
    plt.tight_layout()
    plt.show()
    
    # Plot 2D trajectory
    fig, ax = plt.subplots(figsize=(10,8))
    
    # Plot trajectories and goals first
    pred_traj, = plt.plot(state[:, 0], state[:, 1], 'b', linewidth=5, label='Predicted Trajectory')
    true_traj, = plt.plot(demo_state_traj[:, 0], demo_state_traj[:, 1], 'r--', linewidth=5, label='True Trajectory')
    true_goal_point, = plt.plot(demo_state_traj[-1, 0], demo_state_traj[-1, 1], 'r*', markersize=30, label='True Goal')
    pred_goal_point, = plt.plot(results['theta'][-1][-2], results['theta'][-1][-1], 'b*', markersize=30, label='Predicted Goal')
    
    # Add black rectangle for obstacle last
    obstacle = plt.Rectangle((-0.8, -2), 1.6, 2., color='black', alpha=0.5, label='Obstacle')
    ax.add_patch(obstacle)
    
    plt.xlabel('$x$', fontsize=24)
    plt.ylabel('$y$', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    ax.set_aspect('equal')  # Set equal aspect ratio
    ax.set_ylim(-2.5, 3)  # Set y-axis limits for 2D trajectory

    handles = [true_traj, pred_traj, true_goal_point, pred_goal_point, obstacle]
    labels = ['True Trajectory', 'Predicted Trajectory', 'True Goal', 'Predicted Goal', 'Obstacle']
    ax.legend(handles, labels, fontsize=24, loc='upper right', bbox_to_anchor=(1.1, 1.11))
    
    plt.show()

def plot_learning_curves(results):
    """Plot the learning curves (loss and goal error)"""
    fig, axs = plt.subplots(1, 1, figsize=(10,8))
    
    # Plot goal error
    axs.plot(results['goal_error'][0], 'b', linewidth=5)
    axs.set_ylabel('Prediction Error', fontsize=24)
    axs.set_xlabel('$t$', fontsize=24)
    axs.tick_params(axis='both', which='major', labelsize=24)
    axs.grid(False)
    # axs.set_yscale('log')  # Set y-axis to log base 10
    
    # Add horizontal line at y=0
    axs.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(10,8))
    
    # Plot goal error
    axs.plot(results['Loss'][0], 'b', linewidth=5)
    axs.set_ylabel('Trajectory Loss', fontsize=24)
    axs.set_xlabel('$t$', fontsize=24)
    axs.tick_params(axis='both', which='major', labelsize=24)
    axs.grid(False)
    # Use ScalarFormatter to show ticks as 1, 2, 3... and a 10^x offset
    formatter = ticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Always show 10^x multiplier
    axs.yaxis.set_major_formatter(formatter)
    axs.yaxis.offsetText.set_fontsize(24)
    # axs.set_yscale('log')  # Set y-axis to log base 10
    
    # Add horizontal line at y=0
    axs.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # plt.tight_layout()
    plt.show()

def plot_parameter_convergence(results):
    """Plot the convergence of parameters"""
    theta_history = results['theta']
    true_theta = results['true_theta'][0]
    
    plt.figure(figsize=(10,8))
    for i in range(len(true_theta)):
        plt.plot([theta[i] for theta in theta_history], label=f'θ{i+1}')
        plt.axhline(y=true_theta[i], color='r', linestyle='-', alpha=0.5)
    
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Parameter Value', fontsize=20)
    plt.title('Parameter Convergence', fontsize=24)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.show()

def plot_computation_time(results):
    """Plot the computation time for different components"""
    fig, axs = plt.subplots(3, 1, figsize=(10,8))
    
    # Plot data processing time
    axs[0].plot(results['data_time'][0], 'b')
    axs[0].set_ylabel('Data Processing Time (s)', fontsize=20)
    axs[0].set_title('Computation Time Analysis', fontsize=24)
    axs[0].tick_params(axis='both', which='major', labelsize=18)
    axs[0].grid(True)
    
    # Plot gradient computation time
    axs[1].plot(results['gradient_time'][0], 'r')
    axs[1].set_ylabel('Gradient Computation Time (s)', fontsize=20)
    axs[1].tick_params(axis='both', which='major', labelsize=18)
    axs[1].grid(True)
    
    # Plot EKF update time
    axs[2].plot(results['ekf_time'][0], 'g')
    axs[2].set_ylabel('EKF Update Time (s)', fontsize=20)
    axs[2].set_xlabel('Iterations', fontsize=20)
    axs[2].tick_params(axis='both', which='major', labelsize=18)
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

def create_trajectory_animation(demo_state_traj, results, saveflag=False):
    """Create an animation of the robot's trajectory"""
    print("Demo state shape:", demo_state_traj.shape)
    print("State shape:", results['state'].shape)
    print("Theta shape:", results['theta'].shape)
    
    theta_history = results['theta']  # Get the history of theta values
    print("Theta history shape:", theta_history.shape)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10,8))
    
    # Set up the plot
    ax.set_xlim(min(demo_state_traj[:, 0]) - 2.5,
                max(demo_state_traj[:, 0]) + 2.5)
    ax.set_ylim(-2.5, 2.5)  # Set fixed y-axis limits for animation
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$y$', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # Add black rectangle for obstacle last
    obstacle = plt.Rectangle((-0.5, -2), 1, 2., color='black', alpha=0.5, label='Obstacle')
    ax.add_patch(obstacle)
    
    # Plot trajectories and goals (order matters for z-index)
    demo_line, = ax.plot([], [], 'r--', linewidth=5, label='True Trajectory')  # Plot true trajectory first
    pred_line, = ax.plot([], [], 'b-', linewidth=5, label='Predicted Trajectory')  # Plot predicted trajectory second
    true_goal = results['true_theta'][0][-2:]  # Get the true goal (last two components)
    goal_point = ax.plot(true_goal[0], true_goal[1], 'r*', markersize=30, label='True Goal')[0]
    predicted_goal, = ax.plot([], [], 'b*', markersize=30, label='Predicted Goal')
    
    # Add rectangle for robot last (so it's on top)
    robot_rect = plt.Rectangle((0, 0), 1.0, 0.4, color='red', alpha=0.8, label='Robot')
    ax.add_patch(robot_rect)
    
    # Add time display text
    time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=30)
    # Add computation time display text
    comp_time_text = ax.text(0.02, 0.8, '', transform=ax.transAxes, fontsize=30)
    
    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$y$', fontsize=30)
    ax.set_aspect('equal')  # Set equal aspect ratio
    
    # Add legend
    handles = [demo_line, pred_line, goal_point, predicted_goal, obstacle]
    labels = ['True Trajectory', 'Predicted Trajectory', 'True Goal', 'Predicted Goal', 'Obstacle']
    # ax.legend(handles, labels, fontsize=24, loc='upper right')
    
    def init():
        demo_line.set_data([], [])
        pred_line.set_data([], [])  # Initialize predicted trajectory
        predicted_goal.set_data([], [])
        time_text.set_text('')
        comp_time_text.set_text('')
        robot_rect.set_xy((0, 0))
        robot_rect.set_angle(0)
        return demo_line, pred_line, predicted_goal, time_text, comp_time_text, robot_rect
    
    def update(frame):
        # Update Real Trajectory trajectory
        demo_line.set_data(demo_state_traj[:frame+1, 0], demo_state_traj[:frame+1, 1])
        
        # Update Predicted Trajectory for t > 0
        if frame > 0:
            pred_line.set_data(results['state'][frame, :, 0], results['state'][frame, :, 1])
        else:
            pred_line.set_data(results['state'][frame, :1, 0], results['state'][frame, :1, 1])
        
        # Update robot rectangle
        x, y = demo_state_traj[frame, 0], demo_state_traj[frame, 1]
        yaw = -demo_state_traj[frame, 2]  # Invert the yaw angle
        
        # Calculate offset based on yaw angle
        length = 1.0  # Rectangle length
        width = 0.4   # Rectangle width
        x_offset = -length/2 * np.cos(yaw) + width/2 * np.sin(yaw)
        y_offset = -length/2 * np.sin(yaw) - width/2 * np.cos(yaw)
        
        robot_rect.set_xy((x + x_offset, y + y_offset))  # Apply offset based on yaw
        robot_rect.set_angle(np.degrees(yaw))  # Convert radians to degrees
        
        # Update predicted goal
        current_theta = theta_history[min(frame, theta_history.shape[0]-1)]
        if len(current_theta.shape) > 1:  # If current_theta is 2D array
            current_theta = current_theta.flatten()  # Flatten to 1D array
        predicted_goal.set_data([current_theta[-2]], [current_theta[-1]])
        
        # Update time display (assuming dt = 0.1s)
        current_time = frame * 0.1
        time_text.set_text(f'Time: {current_time:.1f}s')
        
        # Update computation time display
        comp_time = int(results['data_time'][0][min(frame, len(results['data_time'][0])-1)]*1000)
        comp_time_text.set_text(f'Computation Time: {comp_time}ms')
        
        return demo_line, pred_line, predicted_goal, time_text, comp_time_text, robot_rect
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(demo_state_traj),
                        init_func=init, blit=True, interval=100)
    
    # Save animation if saveflag is True
    if saveflag:
        anim.save('dog_animation.mp4', writer='ffmpeg', fps=10, dpi=100)
    
    plt.tight_layout()
    plt.show()
    return anim

def plot_snapshots(demo_state_traj, results):
    """Plot snapshots of the animation at specific time points"""
    # Time points to show (in seconds)
    time_points = [0, 20, 70]
    dt = 0.1  # Time step
    
    for i, t in enumerate(time_points):
        # Calculate frame index
        if t >= len(demo_state_traj):
            t = len(demo_state_traj) - 1
        
        # Create figure for this time point
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get current state
        x, y = demo_state_traj[t, 0], demo_state_traj[t, 1]
        yaw = -demo_state_traj[t, 2]  # Invert the yaw angle
        
        # Plot full predicted trajectory only for t > 0
        if t > 0:
            pred_traj = ax.plot(results['state'][t, :, 0], results['state'][t, :, 1], 'b-', linewidth=5, label='Predicted Trajectory')[0]
        else:
            pred_traj = ax.plot(results['state'][t, :1, 0], results['state'][t, :1, 1], 'b-', linewidth=5, label='Predicted Trajectory')[0]
        
        # Plot trajectories up to current t
        true_traj = ax.plot(demo_state_traj[:t+1, 0], demo_state_traj[:t+1, 1], 'r--', linewidth=5, label='True Trajectory')[0]
        
        # Plot goals
        true_goal = results['true_theta'][0][-2:]
        true_goal_point = ax.plot(true_goal[0], true_goal[1], 'r*', markersize=30, label='True Goal')[0]
        
        current_theta = results['theta'][t]
        if len(current_theta.shape) > 1:
            current_theta = current_theta.flatten()
        pred_goal_point = ax.plot(current_theta[-2], current_theta[-1], 'b*', markersize=30, label='Predicted Goal')[0]
        
        # Add robot rectangle
        length = 1.0
        width = 0.4
        x_offset = -length/2 * np.cos(yaw) + width/2 * np.sin(yaw)
        y_offset = -length/2 * np.sin(yaw) - width/2 * np.cos(yaw)
        robot_rect = plt.Rectangle((x + x_offset, y + y_offset), length, width, angle=np.degrees(yaw), color='red', alpha=0.8, label='Robot')
        ax.add_patch(robot_rect)
        
        # Add obstacle
        obstacle = plt.Rectangle((-0.8, -2), 1.6, 2., color='black', alpha=0.5, label='Obstacle')
        ax.add_patch(obstacle)
        
        # Set plot properties
        ax.set_xlim(-3.5, 4.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel('$x$', fontsize=30)
        ax.set_ylabel('$y$', fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.set_title(f'$t$={int(t)}', fontsize=30)
        ax.set_aspect('equal')  # Set equal aspect ratio
        
        # Add legend only to the last plot
        if i == 0:
            handles = [true_traj, pred_traj, pred_goal_point, true_goal_point, obstacle]
            labels = ['True Trajectory', 'Predicted Trajectory', 'Predicted Goal', 'True Goal', 'Obstacle']
            ax.legend(handles, labels, fontsize=24, loc='upper right')
        
        plt.tight_layout()
        plt.show()

def main():
    try:
        # Load data
        demo_state_traj, demo_traj = load_dog_data()
        results = load_prediction_results()
        print(np.mean(results['data_time']))
        
        # Plot all visualizations
        plot_trajectories(demo_state_traj, results)
        plot_learning_curves(results)
        plot_parameter_convergence(results)
        plot_computation_time(results)
        
        # Plot snapshots
        plot_snapshots(demo_state_traj, results)
        
        # Create and show animation
        anim = create_trajectory_animation(demo_state_traj, results, saveflag=False)  # Set saveflag to True to save animation
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
