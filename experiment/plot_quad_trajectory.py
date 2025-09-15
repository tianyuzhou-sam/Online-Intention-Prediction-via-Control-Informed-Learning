import numpy as np
import matplotlib.pyplot as plt
import csv

# Read the CSV file
with open('experiment/quad1.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    data = []
    for row in csv_reader:
        # Convert string values to float
        data.append([float(val) for val in row])
    data = np.array(data)

# Print data shape and some statistics
print("Data shape:", data.shape)
print("First few rows of data:")
print(data[:5])

# Extract x and y positions (2nd and 3rd columns)
x = data[:, 1]  # 2nd column for x position
y = data[:, 2]  # 3rd column for y position

# Print statistics of x and y
print("\nX position statistics:")
print("Min:", np.min(x))
print("Max:", np.max(x))
print("Mean:", np.mean(x))
print("\nY position statistics:")
print("Min:", np.min(y))
print("Max:", np.max(y))
print("Mean:", np.mean(y))

# Create the plot
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b-', linewidth=2, label='Trajectory')
plt.scatter(x[0], y[0], color='green', s=100, label='Start')
plt.scatter(x[-1], y[-1], color='red', s=100, label='End')

# Add labels and title
plt.xlabel('X Position', fontsize=12)
plt.ylabel('Y Position', fontsize=12)
plt.title('Quadrotor Trajectory', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)

# Set equal aspect ratio and reasonable limits
plt.axis('equal')
# Set reasonable axis limits based on data statistics
x_range = np.max(x) - np.min(x)
y_range = np.max(y) - np.min(y)
plt.xlim(np.min(x) - 0.1*x_range, np.max(x) + 0.1*x_range)
plt.ylim(np.min(y) - 0.1*y_range, np.max(y) + 0.1*y_range)

# Show the plot
plt.show() 