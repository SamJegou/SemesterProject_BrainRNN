import matplotlib.pyplot as plt
import numpy as np

FULL = False
OVERLAY = True
PLOT_STATES = True
PLOT_ACTIONS = True

CLIP_ACTION = True

save_states = 100 # save states every ... episode
filename_suffixe = '_wI09_wG01_in02_out02_std01_gamma097_act'

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

actions = np.load('data/BW_ref_actions2.npy')
states = np.load('data/BW_ref_states2.npy')

only_last_action = False
only_last_state = False
try:
    myactions = np.load('save/actions'+filename_suffixe+'.npy')
except FileNotFoundError:
    myactions = np.load('save/actions'+filename_suffixe+'_last.npy')
    only_last_action = True
try:
    mystates = np.load('save/states'+filename_suffixe+'.npy')
except FileNotFoundError:
    mystates = np.load('save/states'+filename_suffixe+'_last.npy')
    only_last_state = True
mysteps = np.load('save/steps'+filename_suffixe+'.npy')

if not FULL:
    actions = actions[-100:,:]
    states = states[-100:,:]
    last_idx = np.where(mysteps!=0)[0][-1]
    if not only_last_state:
        mystates = mystates[last_idx//save_states,max(0,int(mysteps[last_idx])-100):int(mysteps[last_idx])]
    else:
        mystates = mystates[max(0,int(mysteps[last_idx])-100):int(mysteps[last_idx])]
    if not only_last_action:
        myactions = myactions[last_idx//save_states,max(0,int(mysteps[last_idx])-100):int(mysteps[last_idx])]
    else:
        myactions = myactions[max(0,int(mysteps[last_idx])-100):int(mysteps[last_idx])]

if CLIP_ACTION:
    actions = np.clip(actions, -1,1)
    myactions = np.clip(myactions, -1,1)

t = range(len(states))

if PLOT_STATES:
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

    colors = ['tab:cyan', 'tab:pink']
    if OVERLAY:
        axs[0,0].plot(mystates[:,4], color=colors[0], linestyle='--', label='Leg 1 - train')
        axs[0,0].plot(mystates[:,9], color=colors[1], linestyle='--', label='Leg 2 - train')
        axs[0,0].set_xlabel('Step number')
        axs[0,0].set_ylabel('Hip joint angle')
        axs[0,0].legend()

        axs[1,0].plot(mystates[:,5], color=colors[0], linestyle='--', label='Leg 1 - train')
        axs[1,0].plot(mystates[:,10], color=colors[1], linestyle='--', label='Leg 2 - train')
        axs[1,0].set_xlabel('Step number')
        axs[1,0].set_ylabel('Hip joint speed')
        axs[1,0].legend()

        axs[0,1].plot(mystates[:,6], color=colors[0], linestyle='--', label='Leg 1 - train')
        axs[0,1].plot(mystates[:,11], color=colors[1], linestyle='--', label='Leg 2 - train')
        axs[0,1].set_xlabel('Step number')
        axs[0,1].set_ylabel('Knee joint angle')
        axs[0,1].legend()

        axs[1,1].plot(mystates[:,7], color=colors[0], linestyle='--', label='Leg 1 - train')
        axs[1,1].plot(mystates[:,12], color=colors[1], linestyle='--', label='Leg 2 - train')
        axs[1,1].set_xlabel('Step number')
        axs[1,1].set_ylabel('Knee joint speed')
        axs[1,1].legend()

        axs[2,0].plot(mystates[:,2], color=colors[0], linestyle='--', label='Velocity - train')
        axs[2,0].axhline(np.mean(mystates[:,2]), xmin=0, xmax=1, color=colors[1], linestyle='--', label='Mean - train')
        axs[2,0].set_xlabel('Step number')
        axs[2,0].set_ylabel('Velocity - x')
        axs[2,0].legend()

        axs[2,1].plot(mystates[:,3], color=colors[0], linestyle='--', label='Velocity - train')
        axs[2,1].axhline(np.mean(mystates[:,3]), xmin=0, xmax=1, color=colors[1], linestyle='--', label='Mean - train')
        axs[2,1].set_xlabel('Step number')
        axs[2,1].set_ylabel('Velocity - y')
        axs[2,1].legend()

    if not OVERLAY:
        if FULL:
            plt.suptitle('Reference states')
            plt.savefig('data/reference_states_full')
        else:
            plt.suptitle(f'Reference states (last {len(states)} states)')
            plt.savefig('data/reference_states_partial')
    else:
        if FULL:
            plt.suptitle(f'Reference & trained states ({filename_suffixe})')
            plt.savefig(f'data/reference_states_full_overlay{filename_suffixe}')
        else:
            plt.suptitle(f'Reference & trained states (last {len(states)} states - {filename_suffixe})')
            plt.savefig(f'data/reference_states_partial_overlay{filename_suffixe}')
    plt.show()

if PLOT_ACTIONS:
    fig, axs = plt.subplots(2,2, figsize=(16,9))
    
    colors = ['tab:blue', 'tab:red']
    
    axs[0,0].plot(t, actions[:,0], color=colors[0], label='Leg 1')
    axs[0,0].set_xlabel('Step number')
    axs[0,0].set_ylabel('Hip Torque')
    axs[0,0].legend()

    axs[0,1].plot(t, actions[:,1], color=colors[1], label='Leg 1')
    axs[0,1].set_xlabel('Step number')
    axs[0,1].set_ylabel('Knee Torque')
    axs[0,1].legend()

    axs[1,0].plot(t, actions[:,2], color=colors[0], label='Leg 2')
    axs[1,0].set_xlabel('Step number')
    axs[1,0].set_ylabel('Hip Torque')
    axs[1,0].legend()    

    axs[1,1].plot(t, actions[:,3], color=colors[1], label='Leg 2')
    axs[1,1].set_xlabel('Step number')
    axs[1,1].set_ylabel('Knee Torque')
    axs[1,1].legend()
    
    colors = ['tab:cyan', 'tab:pink']
    if OVERLAY:
        axs[0,0].plot(myactions[:,0], color=colors[0], linestyle='--', label='Leg 1 - train')
        axs[0,0].set_xlabel('Step number')
        axs[0,0].set_ylabel('Hip Torque')
        axs[0,0].legend()

        axs[0,1].plot(myactions[:,1], color=colors[1], linestyle='--', label='Leg 1 - train')
        axs[0,1].set_xlabel('Step number')
        axs[0,1].set_ylabel('Knee Torque')
        axs[0,1].legend()

        axs[1,0].plot(myactions[:,2], color=colors[0], linestyle='--', label='Leg 2 - train')
        axs[1,0].set_xlabel('Step number')
        axs[1,0].set_ylabel('Hip Torque')
        axs[1,0].legend()

        axs[1,1].plot(myactions[:,3], color=colors[1], linestyle='--', label='Leg 2 - train')
        axs[1,1].set_xlabel('Step number')
        axs[1,1].set_ylabel('Knee Torque')
        axs[1,1].legend()

    if CLIP_ACTION:
        for i in range(2):
            for j in range(2):
                axs[i,j].set_ylim(-1.1,1.1)

    if not OVERLAY:
        if FULL:
            plt.suptitle('Reference clipped actions')
            plt.savefig('data/reference_actions_full')
        else:
            plt.suptitle(f'Reference clipped actions (last {len(actions)} actions)')
            plt.savefig(f'data/reference_actions_partial{filename_suffixe}')
    else:
        if FULL:
            plt.suptitle(f'Reference & trained clipped actions ({filename_suffixe})')
            plt.savefig('data/reference_actions_full_overlay')
        else:
            plt.suptitle(f'Reference & trained clipped actions (last {len(actions)} actions - {filename_suffixe})')
            plt.savefig(f'data/reference_actions_partial_overlay{filename_suffixe}')
    plt.show()