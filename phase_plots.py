import matplotlib.pyplot as plt
import numpy as np

save_states = 100 # save states every ... episode
filename_suffixe = '_wI09_wG01_in02_out02_std05'
limb = 'hip' # hip or knee


title = limb+' phase portraits'

x_to_plot = [limb+'_joint_1_angle', limb+'_joint_2_angle', limb+'_joint_1_angle', limb+'_joint_1_speed']
y_to_plot = [limb+'_joint_1_speed', limb+'_joint_2_speed', limb+'_joint_2_angle', limb+'_joint_2_speed']

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

N_steps = 100
ref_states = np.load('data/BW_ref_states2.npy')
states = np.load('save/states'+filename_suffixe+'.npy')
mysteps = np.load('save/steps'+filename_suffixe+'.npy')
last_idx = np.where(mysteps!=0)[0][-1]

x_idx_plot = [obs_space[obs] for obs in x_to_plot]
y_idx_plot = [obs_space[obs] for obs in y_to_plot]

fig, axs = plt.subplots(2,2, figsize=(16,8))
for i in range(2):
    for j in range(2):
        axs[i,j].plot(ref_states[-N_steps:,x_idx_plot[2*i+j]], ref_states[-N_steps:,y_idx_plot[2*i+j]], label='reference', alpha=0.7)
        axs[i,j].plot(states[last_idx//save_states,max(0,int(mysteps[last_idx])-N_steps):int(mysteps[last_idx]),x_idx_plot[2*i+j]],
                      states[last_idx//save_states,max(0,int(mysteps[last_idx])-N_steps):int(mysteps[last_idx]), y_idx_plot[2*i+j]], 
                      linestyle='--', label='train', alpha=0.7)
        axs[i,j].set_xlabel(x_to_plot[2*i+j])
        axs[i,j].set_ylabel(y_to_plot[2*i+j])
        axs[i,j].legend()

plt.suptitle(title+f' (last {N_steps} steps - {filename_suffixe})')
plt.savefig('data/'+title+filename_suffixe)
plt.show()