from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv, CatTensors, UnsqueezeTransform)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torch.distributions import Normal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from torchrl.envs.libs.gym import GymEnv

from settings import *
from BrainRNN import BrainRNN

# Hyperparameters
device = torch.device("cpu")

num_cells = output_size  # number of cells in each layer i.e. output dim.
lr = learning_rate
max_grad_norm = 1.0

# Data collection parameters
frames_per_batch = 10 # nb of interactions with the environment
# For a complete training, bring the number of frames up to 1M
total_frames = 100 # total nb of interactions with environment

# PPO parameters
distribution_scale = 1.
sub_batch_size = batch_size  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4



## Define an environment
base_env = GymEnv("BipedalWalker-v3", device=device)
# Transforms
env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)


# Set normalizatin parameters
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
td = env.reset()
#check_env_specs(env) # Sanity check

obs_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

## Policy
actor_net = BrainRNN(
    obs_dim, 
    a_dim, 
    adj_mat, 
    layers, 
    batch_size=sub_batch_size
)
policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc"]
)
class Distribution(Normal):
    def __init__(self,loc):
        self.loc = loc
        self.scale = distribution_scale
        super().__init__(loc=self.loc, scale=self.scale, validate_args=None)
 
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc"],
    distribution_class=Distribution,
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)
## Value Network
value_net = BrainRNN(
    input_size, 
    output_size, 
    adj_mat, 
    layers, 
    batch_size=sub_batch_size
)
value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)
# print("Running policy:", policy_module(env.reset()))
# print("Running value:", value_module(env.reset()))
## Data collector
for module in [policy_module.module[0].recurrent_layers,
               policy_module.module[0].hidden_layers,
               policy_module.module[0].skip_layers]:
    for i in range(len(module)):
        module[i].weight = module[i].weight.detach()

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)
## Replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)
## Loss function
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

for module in [value_module.module.recurrent_layers,
               value_module.module.hidden_layers,
               value_module.module.skip_layers]:
    for i in range(len(module)):
        module[i].weight = module[i].weight.detach()

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

if __name__=='__main__':
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                # Reset hidden states values
                loss_module.actor_network.module[0].module.reset_hidden_states()
                loss_module.critic_network.reset_hidden_states()

                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()