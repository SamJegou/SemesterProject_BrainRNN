import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers.record_video import RecordVideo
import argparse

from settings import *
from BrainRNN import BrainRNN


### Parameters ###
DUMB_MVT = False
TRAIN = True
CONTINUE_TRAINING = False

device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 300
MIN_SEQUENCE_LEN = 20
N_EPISODE = 20

STD_POLICY = 1.0*2

reset_method = 'random'

DUMB_PATH = 'data/dumb_ref_states.npy'
REF_STATES_PATH = 'data/BW_ref_states2.npy'
if DUMB_MVT:
    ref_states = np.load(DUMB_PATH)
else:
    ref_states = np.load(REF_STATES_PATH)

w_I=0.8
w_G=0.2
w_p=0.7
w_v=0.3

parser = argparse.ArgumentParser(
                prog='RL_bipedal',
                description='Train a BrainRNN RL agent on Bipedal Walker'
                )
parser.add_argument('-N', '--N_episode', default=N_EPISODE, help='total number of episodes during the training')
parser.add_argument('-S', '--Step', default=MAX_STEPS, help='maximal number of steps in an episode')
parser.add_argument('--min_seq_len', default=MIN_SEQUENCE_LEN, help='minimal length of a sequence for learning process')
parser.add_argument('--w_I', default=w_I)
parser.add_argument('--w_G', default=w_G)
parser.add_argument('--w_p', default=w_p)
parser.add_argument('--w_v', default=w_v)
parser.add_argument('-f', '--filename_suffixe',
                    default='',
                    help='suffixe to add to the saved files from this run')
args = parser.parse_args()



joints_angle_idx = [4,6,9,11]
joints_vel_idx = [5,7,10,12]
x_vel_idx = 2
mean_speed = np.mean(ref_states[:,x_vel_idx])

save_dir = 'save'
model_path = os.path.join(save_dir, "model"+args.filename_suffixe+".pt")


#env = GymEnv("BipedalWalker-v3", device=device, render_mode="rgb_array")
env = gym.make("BipedalWalker-v3",
               render_mode="rgb_array",
               max_episode_steps=args.Step)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

env = RecordVideo(env, 
                   video_folder="videoRL/",
                   name_prefix="test-video",
                   episode_trigger=lambda x: x % 10 == 0
                   )

class ModifiedRewardWrapper(Wrapper):
    def __init__(self, env, w_I=0.7, w_G=0.3, w_p=0.65, w_v=0.1, w_e=0.15, w_c=0.1, fall_penalization=20, early_term=False, n_steps_term=40, critic_vel_term=0.1):
        '''
        Args:
        env: gym/torchRL environment
        w_I: float
            imitation reward weight
        w_G: float
            Goal/Task reward weight
        w_p: float
            pose reward weight
        w_v: float
            velocity reward weight
        w_e: float, NOT USED
            end-effector reward weight
        w_c: float, NOT USED
            center of mass reward weight
        fall_penalization: float
            reward penalization if the agent falls
        early_term: bool
            True for early termination if not enough forward movement
        n_steps_term: int
            number of steps on which velocity is averaged to trigger early termination
        critic_vel_term: float
            threshold velocity; under this velocity, early termination will be triggered
        '''
        super().__init__(env)
        imitation_sum = w_p+w_v+w_e+w_c  # normalize weights
        self.w_I = w_I
        self.w_G = w_G
        self.w_v = w_v/imitation_sum
        self.w_p = w_p/imitation_sum
        self.w_e = w_e/imitation_sum
        self.w_c = w_c/imitation_sum
        self.fall_penalization=abs(fall_penalization)
        self.early_term = early_term
        self.n_steps_term = n_steps_term
        self.critic_vel_term = critic_vel_term*self.n_steps_term # to avoid division in mean

        self.step_counter = 0
        self.mean_vel_list = []
        self.mean_vel = 1
    
    def reset(self):
        self.step_counter = 0
        self.mean_vel_list = []
        self.mean_vel = 1
        return self.env.reset()

    def step(self, action):
        obs, reward_base, terminated, truncated, info = self.env.step(action)
        ref = ref_states[self.step_counter,:]

        task_r = np.exp(-2.5*np.max((0, mean_speed - obs[x_vel_idx]))**2)
        if reward_base == -100: # robot fell
            task_r -= self.fall_penalization

        pose_r = np.exp(-2*(
            np.sum((np.cos(ref[joints_angle_idx])-np.cos(obs[joints_angle_idx]))**2)
            + np.sum((np.sin(ref[joints_angle_idx])-np.sin(obs[joints_angle_idx]))**2))
        )
        vel_r = np.exp(-0.1*np.sum((ref[joints_vel_idx]-obs[joints_vel_idx])**2))
        end_effector_r = 0
        center_mass_r = 0
        imitation_r = (self.w_p*pose_r
                       + self.w_v*vel_r
                       + self.w_e*end_effector_r
                       + self.w_c*center_mass_r)

        reward = self.w_I*imitation_r + self.w_G*task_r

        self.step_counter += 1

        if self.early_term:
            if self.step_counter < self.n_steps_term:
                self.mean_vel_list.append(obs[x_vel_idx])
            elif self.step_counter == self.n_steps_term:
                self.mean_vel = np.mean(self.mean_vel_list)
            else:
                old = self.mean_vel_list.pop(0)
                self.mean_vel_list.append(obs[x_vel_idx])
                self.mean_vel += obs[x_vel_idx]-old
            
            if self.mean_vel < self.critic_vel_term:
                terminated = True
                truncated = True
        
        if self.step_counter == len(ref_states)-1:
            truncated = True
            terminated = True
            
        return obs, reward, terminated, truncated, info # terminated = False to stop early termination

class DumbWrapper(Wrapper):
    def __init__(self, env, w_I=1., w_G=0., w_p=1., w_v=0., w_e=0., w_c=0., fall_penalization=0):
        '''
        Args:
        env: gym/torchRL environment
        w_I: float
            imitation reward weight
        w_G: float
            Goal/Task reward weight
        w_p: float
            pose reward weight
        w_v: float
            velocity reward weight
        w_e: float, NOT USED
            end-effector reward weight
        w_c: float, NOT USED
            center of mass reward weight
        fall_penalization: float
            reward penalization if the agent falls
        '''
        super().__init__(env)
        imitation_sum = w_p+w_v+w_e+w_c  # normalize weights
        self.w_I = w_I
        self.w_G = w_G
        self.w_v = w_v/imitation_sum
        self.w_p = w_p/imitation_sum
        self.w_e = w_e/imitation_sum
        self.w_c = w_c/imitation_sum
        self.fall_penalization=abs(fall_penalization)

        self.step_counter = 0
    
    def reset(self):
        self.step_counter = 0
        return self.env.reset()

    def step(self, action):
        obs, reward_base, terminated, truncated, info = self.env.step(action)
        ref = ref_states[self.step_counter,:]

        pose_r = np.exp(-2*(
            np.sum((np.cos(ref[joints_angle_idx])-np.cos(obs[joints_angle_idx]))**2)
            + np.sum((np.sin(ref[joints_angle_idx])-np.sin(obs[joints_angle_idx]))**2))
        )

        imitation_r = self.w_p*pose_r

        reward = self.w_I*imitation_r

        self.step_counter += 1

        terminated = False
        return obs, reward, terminated, truncated, info

### Policy Network & Value Network Construction ###

#AddBias module
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias) #.unsqueeze(1))
    
    def forward(self, x):
        bias = self._bias #.t().view(1, -1)
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
    def __init__(self, inp_dim, out_dim, std=1.0):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(inp_dim, out_dim)
        self.b_logstd = AddBias(torch.zeros(out_dim))
        self.std = std
    
    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = self.b_logstd(torch.zeros_like(mean))
        return FixedNormal(mean, logstd.exp()*self.std)

#Policy Network
class PolicyNet(nn.Module):
    #Constructor
    def __init__(self, s_dim, a_dim, std):
        super(PolicyNet, self).__init__()
        self.main = BrainRNN(s_dim,
                             output_size,
                             adj_mat,
                             layers,
                             batch_size=batch_size)
        self.dist = DiagGaussian(output_size, a_dim, std=std)
    
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

    def reset_hidden_states(self, hidden_states=None, hidden_size=None, method='zero'):
        self.main.reset_hidden_states(hidden_states=hidden_states,
                                      hidden_size=hidden_size,
                                      method=method)

#Value Network
class ValueNet(nn.Module):
    #Constructor
    def __init__(self, s_dim):
        super(ValueNet, self).__init__()
        self.main = BrainRNN(s_dim,
                             1,
                             adj_mat,
                             layers,
                             batch_size=batch_size)
    
    #Forward pass
    def forward(self, state):
        return self.main(state)[..., 0]

    def reset_hidden_states(self, hidden_states=None, hidden_size=None, method='zero'):
        self.main.reset_hidden_states(hidden_states=hidden_states,
                                      hidden_size=hidden_size,
                                      method=method)


### Environment Runner Construction ###

class EnvRunner:
    #Constructor
    def __init__(self, s_dim, a_dim, gamma=0.99, lamb=0.95, max_step=MAX_STEPS, sample_mb_size=batch_size, device='cpu'):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.gamma = gamma
        self.lamb = lamb
        self.max_step = max_step
        self.device = device
        self.sample_mb_size = sample_mb_size

        #Storages (state, action, value, reward, a_logp)
        self.mb_states = np.zeros((self.max_step, self.s_dim), dtype=np.float32)
        self.mb_actions = np.zeros((self.max_step, self.a_dim), dtype=np.float32)
        self.mb_values = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_rewards = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.max_step,), dtype=np.float32)
    
    #Compute discounted return
    def compute_discounted_return(self, rewards, last_value):
        returns = np.zeros_like(rewards)
        n_step = len(rewards)

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                returns[t] = rewards[t] + self.gamma * last_value
            else:
                returns[t] = rewards[t] + self.gamma * returns[t+1]

        return returns
    
    #Compute generalized advantage estimation (Optional)
    def compute_gae(self, rewards, values, last_value):
        advs = np.zeros_like(rewards)
        n_step = len(rewards)
        last_gae_lam = 0.0

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                next_value = last_value
            else:
                next_value = values[t+1]

            delta = rewards[t] + self.gamma*next_value - values[t]
            advs[t] = last_gae_lam = delta + self.gamma*self.lamb*last_gae_lam

        return advs + values

    #Run an episode using the policy net & value net
    def run(self, env, policy_net, value_net):
        #Run an episode
        state, info = env.reset()   #Initial state
        episode_len = self.max_step

        for step in range(self.max_step):
            #state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action, a_logp = policy_net(state_tensor)
            value = value_net(state_tensor)

            action = action.cpu().numpy()
            a_logp = a_logp.cpu().numpy()
            value  = value.cpu().numpy()

            self.mb_states[step] = state
            self.mb_actions[step] = action
            self.mb_a_logps[step] = a_logp
            self.mb_values[step] = value

            state, reward, done, truncated, info = env.step(action)
            self.mb_rewards[step] = reward

            if done:
                episode_len = step + 1
                break
        
        #Compute returns
        last_value = value_net(
            torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
        ).cpu().numpy()

        mb_returns = self.compute_discounted_return(self.mb_rewards[:episode_len], last_value)
        '''
        mb_returns = self.compute_gae(
            self.mb_rewards[:episode_len], 
            self.mb_values[:episode_len],
            last_value
        )
        '''
        return self.mb_states[:episode_len], \
                self.mb_actions[:episode_len], \
                self.mb_a_logps[:episode_len], \
                self.mb_values[:episode_len], \
                mb_returns, \
                self.mb_rewards[:episode_len]

### PPO Algorithm ###

class PPO:
    #Constructor
    def __init__(self, policy_net, value_net, lr=1e-4, max_grad_norm=0.5, ent_weight=0.01, clip_val=0.2, sample_n_epoch=4, sample_mb_size=batch_size, max_seq_len=MIN_SEQUENCE_LEN, device='cpu'):
        self.policy_net = policy_net
        self.value_net = value_net
        self.max_grad_norm = max_grad_norm
        self.ent_weight = ent_weight
        self.clip_val = clip_val
        self.sample_n_epoch = sample_n_epoch
        self.sample_mb_size = sample_mb_size
        self.device = device
        self.opt_polcy = torch.optim.Adam(policy_net.parameters(), lr)
        self.opt_value = torch.optim.Adam(value_net.parameters(), lr)
        self.max_seq_len = max_seq_len # maximal length of a sequence passed in RNN
    
    #Train the policy net & value net using PPO
    def train(self, mb_states, mb_actions, mb_old_values, mb_advs, mb_returns, mb_old_a_logps):
        #Convert numpy array to tensor
        mb_states = torch.from_numpy(mb_states).to(self.device)
        mb_actions = torch.from_numpy(mb_actions).to(self.device)
        mb_old_values = torch.from_numpy(mb_old_values).to(self.device)
        mb_advs = torch.from_numpy(mb_advs).to(self.device)
        mb_returns = torch.from_numpy(mb_returns).to(self.device)
        mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)
        episode_length = len(mb_states)

        k = int(abs(np.log2(episode_length/MIN_SEQUENCE_LEN)))
        sample_n_mb = 2**k
        sequence_length = int(episode_length/sample_n_mb)
        if sequence_length >= episode_length:
            sequence_length = episode_length//2
        #sample_n_mb = (episode_length-self.max_seq_len)//self.sample_mb_size

        #if sample_n_mb <= 0:
        #    sample_n_mb = 1
        #    max_seq_len = episode_length//2
        #else:
        #    max_seq_len = self.max_seq_len


        for i in range(self.sample_n_epoch):
            #rand_idx = np.random.choice(episode_length-max_seq_len, size=sample_n_mb*self.sample_mb_size)
            sample_idx = np.random.choice(episode_length-sequence_length, size=sample_n_mb)

            
            #Randomly sample a batch for training
            v_loss, pg_loss = 0, 0
            self.policy_net.reset_hidden_states(hidden_size=sample_n_mb, method=reset_method)
            self.value_net.reset_hidden_states(hidden_size=sample_n_mb, method=reset_method)
            for t in range(sequence_length):
                sample_states = mb_states[sample_idx]
                sample_actions = mb_actions[sample_idx]
                sample_old_values = mb_old_values[sample_idx]
                sample_advs = mb_advs[sample_idx]
                sample_returns = mb_returns[sample_idx]
                sample_old_a_logps = mb_old_a_logps[sample_idx]
                sample_a_logps, sample_ents = self.policy_net.evaluate(sample_states, sample_actions)
                sample_values = self.value_net(sample_states)
                ent = sample_ents.mean()
                #Compute value loss
                v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.clip_val, self.clip_val)
                v_loss1 = (sample_returns - sample_values).pow(2)
                v_loss2 = (sample_returns - v_pred_clip).pow(2)
                v_loss += torch.max(v_loss1, v_loss2).mean()
                #Compute policy gradient loss
                ratio = (sample_a_logps - sample_old_a_logps).exp()
                pg_loss1 = -sample_advs * ratio
                pg_loss2 = -sample_advs * torch.clamp(ratio, 1.0-self.clip_val, 1.0+self.clip_val)
                pg_loss += torch.max(pg_loss1, pg_loss2).mean() - self.ent_weight*ent
                sample_idx += 1

            #Train actor
            self.opt_polcy.zero_grad()
            pg_loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.opt_polcy.step()
            #Train critic
            self.opt_value.zero_grad()
            v_loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.opt_value.step()

        return pg_loss.item(), v_loss.item(), ent.item()


### Training and Testing Process ###
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
            action = policy_net.choose_action(state_tensor, deterministic=False).cpu().numpy()
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

def train(env, runner, policy_net, value_net, agent, max_episode=args.N_episode):
    mean_total_reward = 0
    mean_length = 0
    save_dir = 'train'

    all_rewards = np.zeros(max_episode)
    all_steps = np.zeros(max_episode)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(max_episode):
        #Run an episode to collect data
        with torch.no_grad():
            policy_net.reset_hidden_states(hidden_size=0, method=reset_method)
            value_net.reset_hidden_states(hidden_size=0, method=reset_method)
            mb_states, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = runner.run(env, policy_net, value_net)
            mb_advs = mb_returns - mb_values
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

        all_rewards[i] = mb_rewards.sum()
        all_steps[i] = len(mb_states)

        #Train the model using the collected data
        policy_net.reset_hidden_states(method=reset_method)
        value_net.reset_hidden_states(method=reset_method) # need batched data as agent.train evaluates BrainRNN on [B,...] samples from the episode
        pg_loss, v_loss, ent = agent.train(mb_states, mb_actions, mb_values, mb_advs, mb_returns, mb_old_a_logps)
        mean_total_reward += mb_rewards.sum()
        mean_length += len(mb_states)
        print("[Episode {:4d}] total reward = {:.6f}, length = {:d}".format(i, mb_rewards.sum(), len(mb_states)))

        #Show the current result & save the model
        if i % 200 == 0:
            print("\n[{:5d} / {:5d}]".format(i, max_episode))
            print("----------------------------------")
            print("actor loss = {:.6f}".format(pg_loss))
            print("critic loss = {:.6f}".format(v_loss))
            print("entropy = {:.6f}".format(ent))
            print("mean return = {:.6f}".format(mean_total_reward / 200))
            print("mean length = {:.2f}".format(mean_length / 200))
            print("\nSaving the model ... ", end="")
            torch.save({
                "it": i,
                "PolicyNet": policy_net.state_dict(),
                "ValueNet": value_net.state_dict()
            }, os.path.join(save_dir, "model"+args.filename_suffixe+".pt"))
            print("Done.")
            print()
            #play(policy_net)
            mean_total_reward = 0
            mean_length = 0
        
    return all_rewards, all_steps

if __name__ == '__main__':
    ### Training/Evaluation
    policy_net = PolicyNet(s_dim, a_dim, std=STD_POLICY)
    value_net = ValueNet(s_dim)
    runner = EnvRunner(s_dim, a_dim)
    agent = PPO(policy_net, value_net)

    if TRAIN:
        if DUMB_MVT:
            env = DumbWrapper(env)
        else:
            env = ModifiedRewardWrapper(env, 
                                   w_I=args.w_I, 
                                   w_G=args.w_G, 
                                   fall_penalization=5,
                                   w_p=args.w_p, 
                                   w_v=args.w_v,
                                   early_term=False,
                                   n_steps_term=40,
                                   critic_vel_term=0)
        if CONTINUE_TRAINING: # load precedent saved models
            if os.path.exists(model_path):
                print("Loading the model ... ", end="")
                checkpoint = torch.load(model_path)
                policy_net.load_state_dict(checkpoint["PolicyNet"])
                value_net.load_state_dict(checkpoint["ValueNet"])
                print("Done.")
            else:
                print('ERROR: No model saved')

        rewards, steps = train(env, runner, policy_net, value_net, agent)

        torch.save({
            "PolicyNet": policy_net.state_dict(),
            "ValueNet": value_net.state_dict()
        }, os.path.join('save', "model.pt"))
        env.close()

        np.save('train/rewards'+args.filename_suffixe+'.npy', rewards)
        np.save('train/steps'+args.filename_suffixe+'.npy', steps)

    else:
        if os.path.exists(model_path):
            print("Loading the model ... ", end="")
            checkpoint = torch.load(model_path)
            policy_net.load_state_dict(checkpoint["PolicyNet"])
            print("Done.")
        else:
            print('ERROR: No model saved')

        #policy_net.main.batch_size = 0
        #hidden_states = policy_net.main.hidden_states[5,:]
        policy_net.reset_hidden_states(hidden_size=0, method=reset_method)
        play(policy_net)