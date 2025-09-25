import matplotlib.pyplot as plt 
import scipy.io as sio
import numpy as np
import math


iter = 100
T_memory = 10
H = 120

dt = 20
loss_20 = list()
for i in range(iter):
    data = sio.loadmat('results/20/result_' + str(i) + '.mat')
    loss_each = data['goal_error'][0]
    if i == 0:
        loss_20 = loss_each
    else:
        loss_20 = np.vstack((loss_20, loss_each))

avg_loss_20 = np.mean(loss_20, 0)


error_20 = []
for i in range(iter):
    for t in range(int(H/dt)+1):
        error_each = loss_20[i, dt*t-2]
        error_20.append(error_each)

avg_error_20 = np.mean(error_20, 0)
std_20 = np.std(error_20)


dt = 30
loss_30 = list()
for i in range(iter):
    data = sio.loadmat('results/30/result_' + str(i) + '.mat')
    loss_each = data['goal_error'][0]
    if i == 0:
        loss_30 = loss_each
    else:
        loss_30 = np.vstack((loss_30, loss_each))

avg_loss_30 = np.mean(loss_30, 0)

error_30 = []
for i in range(iter):
    for t in range(int(H/dt)+1):
        error_each = loss_30[i, dt*t-2]
        error_30.append(error_each)

avg_error_30 = np.mean(error_30, 0)
std_30 = np.std(error_30)


dt = 40
loss_40 = list()
for i in range(iter):
    data = sio.loadmat('results/40/result_' + str(i) + '.mat')
    loss_each = data['goal_error'][0]
    if i == 0:
        loss_40 = loss_each
    else:
        loss_40 = np.vstack((loss_40, loss_each))
avg_loss_40 = np.mean(loss_40, 0)

error_40 = []
for i in range(iter):
    for t in range(int(H/dt)+1):
        error_each = loss_40[i, dt*t-2]
        error_40.append(error_each)
avg_error_40 = np.mean(error_40, 0)
std_40 = np.std(error_40)


dt = 60
loss_60 = list()
for i in range(iter):
    data = sio.loadmat('results/60/result_' + str(i) + '.mat')
    loss_each = data['goal_error'][0]
    if i == 0:
        loss_60 = loss_each
    else:
        loss_60 = np.vstack((loss_60, loss_each))

avg_loss_60 = np.mean(loss_60, 0)

error_60 = []
for i in range(iter):
    for t in range(int(H/dt)+1):
        error_each = loss_60[i, dt*t-2]
        error_60.append(error_each)
avg_error_60 = np.mean(error_60, 0)
std_60 = np.std(error_60)

dt = 20
loss_20_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/20_OCIL/result_' + str(i) + '.mat')
    loss_each = data['goal_error'][0]
    if i == 0:
        loss_20_OCIL = loss_each
    else:
        loss_20_OCIL = np.vstack((loss_20_OCIL, loss_each))
avg_loss_20_OCIL = np.mean(loss_20_OCIL, 0)

error_20_OCIL = []
for i in range(iter):
    for t in range(int(H/dt)+1):
        error_each = loss_20_OCIL[i, dt*t-2]
        error_20_OCIL.append(error_each)
avg_error_20_OCIL = np.mean(error_20_OCIL, 0)
std_20_OCIL = np.std(error_20_OCIL)

dt = 30
loss_30_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/30_OCIL/result_' + str(i) + '.mat')
    loss_each = data['goal_error'][0]
    if i == 0:
        loss_30_OCIL = loss_each
    else:
        loss_30_OCIL = np.vstack((loss_30_OCIL, loss_each))
avg_loss_30_OCIL = np.mean(loss_30_OCIL, 0)

error_30_OCIL = []
for i in range(iter):
    for t in range(int(H/dt)+1):
        error_each = loss_30_OCIL[i, dt*t-2]
        error_30_OCIL.append(error_each)
avg_error_30_OCIL = np.mean(error_30_OCIL, 0)
std_30_OCIL = np.std(error_30_OCIL)


dt = 40
loss_40_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/40_OCIL/result_' + str(i) + '.mat')
    loss_each = data['goal_error'][0]
    if i == 0:
        loss_40_OCIL = loss_each
    else:
        loss_40_OCIL = np.vstack((loss_40_OCIL, loss_each))
avg_loss_40_OCIL = np.mean(loss_40_OCIL, 0)

error_40_OCIL = []
for i in range(iter):
    for t in range(int(H/dt)+1):
        error_each = loss_40_OCIL[i, dt*t-2]
        error_40_OCIL.append(error_each)
avg_error_40_OCIL = np.mean(error_40_OCIL, 0)
std_40_OCIL = np.std(error_40_OCIL)


dt = 60
loss_60_OCIL = list()
for i in range(iter):
    data = sio.loadmat('results/60_OCIL/result_' + str(i) + '.mat')
    loss_each = data['goal_error'][0]
    if i == 0:
        loss_60_OCIL = loss_each
    else:
        loss_60_OCIL = np.vstack((loss_60_OCIL, loss_each))
avg_loss_60_OCIL = np.mean(loss_60_OCIL, 0)

error_60_OCIL = []
for i in range(iter):
    for t in range(int(H/dt)+1):
        error_each = loss_60_OCIL[i, dt*t-2]
        error_60_OCIL.append(error_each)
avg_error_60_OCIL = np.mean(error_60_OCIL, 0)
std_60_OCIL = np.std(error_60_OCIL)


# Create bar plot with error bars
fig, ax = plt.subplots(figsize=(6, 6))

# Data for plotting - cases ordered from 20 to 60 (excluding 10)
cases = [20, 30, 40, 60]
means_data = [
    [avg_error_20, avg_error_20_OCIL], 
    [avg_error_30, avg_error_30_OCIL],
    [avg_error_40, avg_error_40_OCIL],
    [avg_error_60, avg_error_60_OCIL]
]
errors_data = [
    [std_20, std_20_OCIL],
    [std_30, std_30_OCIL],
    [std_40, std_40_OCIL],
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
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, mean, error) in enumerate(zip(all_bars, all_means, all_errors)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + error + height*0.01,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Proposed'),
                  Patch(facecolor='green', alpha=0.7, label='OCIL')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=20)

ax.set_ylim(0, None)

plt.tight_layout()
plt.show()

# Create second bar plot without error bars
fig2, ax2 = plt.subplots(figsize=(6, 6))

# Data for plotting - cases ordered from 20 to 60 (excluding 10)
cases2 = [20, 30, 40, 60]
means_data2 = [
    [avg_error_20, avg_error_20_OCIL], 
    [avg_error_30, avg_error_30_OCIL],
    [avg_error_40, avg_error_40_OCIL],
    [avg_error_60, avg_error_60_OCIL]
]

colors2 = ['blue', 'green']  # Blue for Proposed, Green for OCIL
bar_width2 = 0.35

# Create bars for all cases without error bars
all_bars2 = []
all_means2 = []

for i, case in enumerate(cases2):
    x_pos = i
    means = means_data2[i]
    
    bars = ax2.bar([x_pos - bar_width2/2, x_pos + bar_width2/2], means, 
                   alpha=0.7, width=bar_width2,
                   color=colors2, edgecolor='black', linewidth=1)
    
    all_bars2.extend(bars)
    all_means2.extend(means)

# Customize the second plot
ax2.set_xlabel('$\Delta$', fontsize=20)
ax2.set_ylabel('Average Prediction Loss', fontsize=20)
ax2.set_xticks(range(len(cases2)))
ax2.set_xticklabels([str(case) for case in cases2], fontsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, mean) in enumerate(zip(all_bars2, all_means2)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add legend
legend_elements2 = [Patch(facecolor='blue', alpha=0.7, label='Proposed'),
                   Patch(facecolor='green', alpha=0.7, label='OCIL')]
ax2.legend(handles=legend_elements2, loc='upper right', fontsize=20)

ax2.set_ylim(0, None)

plt.tight_layout()
plt.show()



fig3, ax3 = plt.subplots(2,1, figsize=(6, 6), sharey=True, gridspec_kw={'hspace':0.5})

ax3[0].plot(avg_loss_20, color='b', linewidth=4)
ax3[0].plot(avg_loss_20_OCIL, color='g', linestyle='--', linewidth=4)
ax3[0].set_xlabel('$t$', fontsize=16)
# ax3[0].legend(['Proposed', 'OCIL'], fontsize=20)
ax3[0].set_title('$\Delta$ = 20', fontsize=20)
ax3[0].tick_params(axis='both', labelsize=16)
ax3[0].grid(True, alpha=0.3, axis='y')
ax3[0].set_xlim(0, 120)

ax3[1].plot(avg_loss_30, color='b', linewidth=4)
ax3[1].plot(avg_loss_30_OCIL, color='g', linestyle='--', linewidth=4)
ax3[1].set_xlabel('$t$', fontsize=16)
ax3[1].legend(['Proposed', 'OCIL'],
              fontsize=16,
              loc='upper left',          # anchor corner
              bbox_to_anchor=(-0.05, 1.4))  # x,y relative to axes

ax3[1].set_title('$\Delta$ = 30', fontsize=20)
ax3[1].tick_params(axis='both', labelsize=16)
ax3[1].grid(True, alpha=0.3, axis='y')
ax3[1].set_xlim(0, 120)

# Add shared ylabel
fig3.text(-0, 0.5, 'Prediction Loss', va='center', rotation='vertical', fontsize=20)


plt.tight_layout()
plt.show()


# import seaborn as sns
loss_matrix = np.vstack((avg_loss_20, avg_loss_30, avg_loss_40, avg_loss_60))
loss_matrix_OCIL = np.vstack((avg_loss_20_OCIL, avg_loss_30_OCIL, avg_loss_40_OCIL, avg_loss_60_OCIL))
t = list(range(0, 120))
# Deltas = [
#     '20\n(Proposed)',
#     '20\n(OCIL)',
#     '30\n(Proposed)',
#     '30\n(OCIL)',
#     '40\n(Proposed)',
#     '40\n(OCIL)',
#     '60\n(Proposed)',
#     '60\n(OCIL)'
# ]
# # ax = sns.heatmap(loss_matrix, xticklabels=t, yticklabels=Deltas)
# # ax.set_xlabel('$t$', fontsize=20)
# # ax.set_ylabel('$\Delta$', fontsize=20)
# # ax.tick_params(axis='both', labelsize=20)
# # plt.tight_layout()
# # plt.show()

# font_size = 20
# plt.rcParams.update({
#     'font.size': 20,         # default text size
#     'axes.titlesize': 20,    # title font size
#     'axes.labelsize': 20,    # x/y label size
#     'xtick.labelsize': 20,   # x tick label size
#     'ytick.labelsize': 20,   # y tick label size
#     'legend.fontsize': 20,   # legend font size
#     'figure.titlesize': 20   # figure title size
# })
loss_matrix_all = np.vstack((avg_loss_20, avg_loss_20_OCIL, avg_loss_30, avg_loss_30_OCIL, avg_loss_40, avg_loss_40_OCIL, avg_loss_60, avg_loss_60_OCIL))
# fig, ax = plt.subplots(figsize=(12,6))
# im = ax.imshow(loss_matrix_all, aspect='auto',
#                origin='lower',  # so first Δ at bottom
#                cmap='viridis')  # or 'plasma', 'inferno', etc.
# # Add colorbar
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label('Average Prediction Loss')
# # Set axis labels and ticks
# ax.set_xlabel('$t$')
# ax.set_ylabel('$\Delta$')
# ax.set_yticks(np.arange(len(Deltas)))
# ax.set_yticklabels(Deltas, ha='center')
# # Show fewer xticks if too many
# step = 20
# ax.set_xticks(np.arange(0, len(t), step))
# ax.set_xticklabels(t[::step])
# ax.tick_params(axis='y', which='major', pad=20)
# # plt.title('Average Prediction Loss Heatmap')
# plt.tight_layout()
# plt.show()


# Deltas = ['20', '30', '40', '60']
# fig, ax = plt.subplots(figsize=(8,8))
# im = ax.imshow(loss_matrix, aspect='auto',
#                origin='lower',  # so first Δ at bottom
#                cmap='viridis')  # or 'plasma', 'inferno', etc.
# # Add colorbar
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label('Average Prediction Loss')
# # Set axis labels and ticks
# ax.set_xlabel('$t$')
# ax.set_ylabel('$\Delta$')
# ax.set_yticks(np.arange(len(Deltas)))
# ax.set_yticklabels(Deltas)
# # Show fewer xticks if too many
# step = 20
# ax.set_xticks(np.arange(0, len(t), step))
# ax.set_xticklabels(t[::step])
# # plt.title('Average Prediction Loss Heatmap')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(8,8))
# im = ax.imshow(loss_matrix_OCIL, aspect='auto',
#                origin='lower',  # so first Δ at bottom
#                cmap='viridis')  # or 'plasma', 'inferno', etc.
# # Add colorbar
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label('Average Prediction Loss')
# # Set axis labels and ticks
# ax.set_xlabel('$t$')
# ax.set_ylabel('$\Delta$')
# ax.set_yticks(np.arange(len(Deltas)))
# ax.set_yticklabels(Deltas)
# # Show fewer xticks if too many
# step = 20
# ax.set_xticks(np.arange(0, len(t), step))
# ax.set_xticklabels(t[::step])
# # plt.title('Average Prediction Loss Heatmap')
# plt.tight_layout()
# plt.show()


