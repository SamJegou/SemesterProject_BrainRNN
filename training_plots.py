import matplotlib.pyplot as plt
import numpy as np

filename_suffixe = '_wI07_WG03'
obs_to_plot = ['hip_joint_1_angle', 'hip_joint_2_angle']

obs_space = {
    'hull_angle':0,
    'hull_angular_vel':1,
    'vel_x':2,
    'vel_y':3,
    'hip_joint_1_angle':4,
    'hip_joint_1_speed':5,
    'knee_joint_1_angle':6,
    'knee_joint_1_speed':7,
    'leg_ground_contact_flag':8,
    'hip_joint_2_angle':9,
    'hip_joint_2_speed':10,
    'knee_joint_2_angle':11,
    'knee_joint_2_speed':12,
    'leg_ground_contact_flag':13,
    'lidar_readings':list(range(14,24))
}
rewards = np.load('train/rewards'+filename_suffixe+'.npy')
steps = np.load('train/steps'+filename_suffixe+'.npy')
states = np.load('train/states'+filename_suffixe+'.npy')

t = range(len(rewards))
idx_states = [obs_space[obs] for obs in obs_to_plot]

# Reward & steps
fig, axs = plt.subplots(1,2)

color = 'tab:blue'
axs[0].set_ylabel('Step number', color=color)  # we already handled the x-label with ax1
axs[0].plot(t, steps, color=color, linestyle='--', alpha=0.5)
axs[0].tick_params(axis='y', labelcolor=color)

ax2 = axs[0].twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:red'
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Reward', color=color)
ax2.scatter(t, rewards, color=color)
ax2.tick_params(axis='y', labelcolor=color)

for i,idx in enumerate(idx_states):
    axs[1].plot(range(states.shape[1]), states[-1,:,idx], label=obs_to_plot[i])
axs[1].set_xlabel('Step')
axs[1].set_title('Observation variables (last epoch)')
axs[1].legend()

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Training'+filename_suffixe)
plt.grid()
plt.show()

