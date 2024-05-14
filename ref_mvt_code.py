import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from gymnasium.wrappers.record_video import RecordVideo

from RL_bipedal_hand import ModifiedRewardWrapper


DUMB_MVT = False
DUMB_PATH = 'data/dumb_ref_states.npy'
freq = 1.

model_path = 'models/ref_mvt2.pt'
REF_STATES_PATH = 'data/BW_ref_states2.npy'

MAX_STEPS = 1600
env = gym.make("BipedalWalker-v3",
               render_mode="rgb_array",
               max_episode_steps=MAX_STEPS)

env = RecordVideo(env, 
                   video_folder="videoRL/",
                   name_prefix="reference",
                   episode_trigger=lambda x: x % 5 == 0)
env = ModifiedRewardWrapper(env, 
                            w_I=0.8, 
                            w_G=0.2, 
                            fall_penalization=5,
                            w_p=0.5, 
                            w_v=0.5,
                            early_term=False,
                            n_steps_term=40,
                            critic_vel_term=0
                            )
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

def play(policy_net):

    with torch.no_grad():
        state, info = env.reset()
        total_reward = 0
        length = 0

        while True:
            env.render()
            #state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device='cpu')
            state_tensor = torch.tensor(state, dtype=torch.float32, device='cpu')
            #state_tensor = torch.tensor(state.expand(1), dtype=torch.float32, device='cpu')
            action = policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()
            if action.ndim > 1:
                state, reward, done, truncated, info = env.step(action[0])
            else:
                state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            length += 1

            if done or truncated:
                print("[Evaluation] Total reward = {:.6f}, length = {:d}".format(total_reward, length), flush=True)
                break
    #env.play()
    env.close()

def save_states(policy_net, save_path=REF_STATES_PATH):

    with torch.no_grad():
        state, info = env.reset()
        total_reward = 0
        length = 0

        res = state

        while True:
            #state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device='cpu')
            state_tensor = torch.tensor(state, dtype=torch.float32, device='cpu')
            #state_tensor = torch.tensor(state.expand(1), dtype=torch.float32, device='cpu')
            action = policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()
            if action.ndim > 1:
                state, reward, done, truncated, info = env.step(action[0])
            else:
                state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            length += 1

            res = np.vstack((res, state))
            if done or truncated:
                print("[Evaluation] Total reward = {:.6f}, length = {:d}".format(total_reward, length), flush=True)
                with open(save_path, 'wb') as file:
                    np.save(file, res)
                print('Reference states saved')
                break
    #env.play()
    env.close()

#AddBias module
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        bias = self._bias.t().view(1, -1)
        return x + bias

#Gaussian distribution with given mean & std.
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, x):
        return super().log_prob(x).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

#Diagonal Gaussian module
class DiagGaussian(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(inp_dim, out_dim)
        self.b_logstd = AddBias(torch.zeros(out_dim))

    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = self.b_logstd(torch.zeros_like(mean))
        return FixedNormal(mean, logstd.exp())
    
class PolicyNet(nn.Module):
    #Constructor
    def __init__(self, s_dim, a_dim):
        super(PolicyNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.dist = DiagGaussian(128, a_dim)

    #Forward pass
    def forward(self, state, deterministic=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action, dist.log_probs(action)

    #Choose an action (stochastically or deterministically)
    def choose_action(self, state, deterministic=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            return dist.mode()

        return dist.sample()

    #Evaluate a state-action pair (output log-prob. & entropy)
    def evaluate(self, state, action):
        feature = self.main(state)
        dist = self.dist(feature)
        return dist.log_probs(action), dist.entropy()

if __name__ == '__main__':
    if DUMB_MVT:
        res = np.zeros((510, 24))
        omega_t = 2*np.pi*freq*np.linspace(0,510*4//200, 510) # 200 steps <-> 4sec
        joints_angle_idx = [4,6,9,11]
        joints_vel_idx = [5,7,10,12]
        res[:,6] = omega_t
        res[100:,11] = 0

        np.save(DUMB_PATH, res)
    else:
        policy_net = PolicyNet(s_dim, a_dim)
        if os.path.exists(model_path):
            print("Loading the model ... ", end="")
            checkpoint = torch.load(model_path)
            policy_net.load_state_dict(checkpoint["PolicyNet"])
            print("Done.")
        else:
            print('ERROR: No model saved')

        #play(policy_net)
        save_states(policy_net)
    