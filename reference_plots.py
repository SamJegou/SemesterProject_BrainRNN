import matplotlib.pyplot as plt
import numpy as np

FULL = True

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

states = np.load('data/BW_ref_states2.npy')

if not FULL:
    states = states[-100:,:]
t = range(len(states))

# Reward & steps
fig, axs = plt.subplots(3,2, figsize=(16,9))

colors = ['tab:blue', 'tab:red']

axs[0,0].plot(t, states[:,4], color=colors[0], label='Leg 1')
axs[0,0].plot(t, states[:,9], color=colors[1], label='Leg 2')
axs[0,0].set_xlabel('Step number')
axs[0,0].set_ylabel('Hip joint angle')
axs[0,0].legend()

axs[1,0].plot(t, states[:,5], color=colors[0], label='Leg 1')
axs[1,0].plot(t, states[:,10], color=colors[1], label='Leg 2')
axs[1,0].set_xlabel('Step number')
axs[1,0].set_ylabel('Hip joint speed')
axs[1,0].legend()

axs[0,1].plot(t, states[:,6], color=colors[0], label='Leg 1')
axs[0,1].plot(t, states[:,11], color=colors[1], label='Leg 2')
axs[0,1].set_xlabel('Step number')
axs[0,1].set_ylabel('Knee joint angle')
axs[0,1].legend()

axs[1,1].plot(t, states[:,7], color=colors[0], label='Leg 1')
axs[1,1].plot(t, states[:,12], color=colors[1], label='Leg 2')
axs[1,1].set_xlabel('Step number')
axs[1,1].set_ylabel('Knee joint speed')
axs[1,1].legend()

axs[2,0].plot(t, states[:,2], color=colors[0], label='Velocity')
axs[2,0].axhline(np.mean(states[:,2]), xmin=0, xmax=1, color=colors[1], label='Mean')
axs[2,0].set_xlabel('Step number')
axs[2,0].set_ylabel('Velocity - x')
axs[2,0].legend()

axs[2,1].plot(t, states[:,3], color=colors[0], label='Velocity')
axs[2,1].axhline(np.mean(states[:,3]), xmin=0, xmax=1, color=colors[1], label='Mean')
axs[2,1].set_xlabel('Step number')
axs[2,1].set_ylabel('Velocity - y')
axs[2,1].legend()

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
if FULL:
    plt.suptitle('Reference states')
    plt.savefig('data/reference_states_full')
else:
    plt.suptitle(f'Reference states (last {len(states)} states)')
    plt.savefig('data/reference_states_partial')
plt.show()


