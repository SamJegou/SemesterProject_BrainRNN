import matplotlib.pyplot as plt
import numpy as np

FULL = True

save_states = 100 # save states every ... episode
filename_suffixe = '_wI09_wG01_in02_out02_std01_act'
obs_to_plot = ['hip_joint_1_angle', 'hip_joint_2_angle'] + ['knee_joint_1_angle', 'knee_joint_2_angle']

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

rewards = np.load('save/rewards'+filename_suffixe+'.npy')
steps = np.load('save/steps'+filename_suffixe+'.npy')
states = np.load('save/states'+filename_suffixe+'.npy')
pg_loss = np.load('save/pg_loss'+filename_suffixe+'.npy')
v_loss = np.load('save/v_loss'+filename_suffixe+'.npy')


if not FULL:
    states = states[-100:,:]
    rewards = rewards[-100:]
    steps = steps[-100:]
    pg_loss = pg_loss[-100:]
    v_loss = v_loss [-100:]

t = range(len(rewards))
idx_states = [obs_space[obs] for obs in obs_to_plot]

# Reward & steps
fig, axs = plt.subplots(1,3, figsize=(16,8))

color = 'tab:red'
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Reward', color=color)  # we already handled the x-label with ax1
axs[0].scatter(t, rewards, s=10, color=color)
axs[0].grid()

axs[1].scatter(t, pg_loss, s=10, label='pg_loss')
axs[1].scatter(t, v_loss, s=10, label='v_loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].set_title('Loss evolution')
axs[1].set_yscale('log')
axs[1].legend()
axs[1].grid()

last_idx = np.where(steps!=0)[0][-1]
for i,idx in enumerate(idx_states):
    axs[2].plot(range(states.shape[1]), states[last_idx//save_states,:,idx], label=obs_to_plot[i])
axs[2].set_xlabel('Step')
axs[2].set_ylabel('Angle (rad)')
axs[2].set_title('Observation variables (last epoch)')
axs[2].legend()
axs[2].grid()

if FULL:
    plt.suptitle('Training'+filename_suffixe)
    plt.savefig(f'data/{filename_suffixe}_states_full')
else:
    plt.suptitle('Training'+filename_suffixe+f'(last {len(states)} states)')
    plt.savefig(f'data/{filename_suffixe}_states_partial')
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

