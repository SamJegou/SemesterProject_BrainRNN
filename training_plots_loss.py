import matplotlib.pyplot as plt
import numpy as np

REF_STATES_PATH = 'data/BW_ref_states2.npy'
DEB = 0
END = 2000
save_states = 100

save_states = 100 # save states every ... episode
filename_suffixe = '_wI09_wG01_in02_out02_std1_act'
obs_to_plot = ['hip_joint_1_angle', 'hip_joint_2_angle']
joints_angle_idx = [4,6,9,11]
joints_vel_idx = [5,7,10,12]

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

ref_states = np.load(REF_STATES_PATH)
rewards = np.load('save/rewards'+filename_suffixe+'.npy')
steps = np.load('save/steps'+filename_suffixe+'.npy')
states = np.load('save/states'+filename_suffixe+'.npy')

rewards = rewards[DEB:END]

angle_loss = np.zeros(len(states))
speed_loss = np.zeros(len(states))
for i in range(len(states)):
    pose_r = 0
    vel_r = 0
    for j in range(int(steps[save_states*i])):
        pose_r += np.exp(-1*(
                    np.sum((np.cos(ref_states[j, joints_angle_idx])-np.cos(states[i][j, joints_angle_idx]))**2)
                    + np.sum((np.sin(ref_states[j, joints_angle_idx])-np.sin(states[i][j, joints_angle_idx]))**2))
                )
        vel_r += np.exp(-2*np.sum((ref_states[j, joints_vel_idx]-states[i][j, joints_vel_idx])**2))
    
    angle_loss[i] = pose_r
    speed_loss[i] = vel_r
    

t = np.linspace(0, len(states)*save_states, len(states))

plt.title(filename_suffixe)
plt.plot(t[:-9], angle_loss[:-9], label='angular reward')
plt.plot(t[:-9], speed_loss[:-9], label='angular speed reward')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.legend()
plt.show()

