from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

from settings import Kp, Kd, time_step, adj_mat, nb_joints, batch_size

def update_pos(angles, lengths):
    c = torch.cos(angles)
    s = torch.sin(angles)
    rel_pos = torch.stack([lengths*c, lengths*s], dim=-1) # (batch, num joint, cos/sin)
    pos = torch.cumsum(rel_pos, dim=-2, dtype=torch.float32) #(batch, num_joint, x/y)
    return pos

def pd_controller(epsilon, epsilon_dot, Kp=Kp, Kd=Kd):
    return Kp*epsilon+Kd*epsilon_dot

def imitation_reward(th, thdot, th_target, thdot_target, w_p=0.85, w_v=0.25):
    pose = torch.exp(-2*torch.sum((th-th_target)**2, dim=-1))
    vel = torch.exp(-0.1*torch.sum((thdot-thdot_target)**2, dim=-1))
    return w_p*pose + w_v*vel

def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

class FlyEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False #will not enforce the input tensordict to have a batch-size that matches the one of the environment
    
    def __init__(self, nb_joints=nb_joints, td_params=None, seed=None, device="cpu"):
        self.nb_joints = nb_joints
        self.dt = time_step
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[batch_size])
        self._make_spec(td_params)
        
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def gen_params(self, batch_size=batch_size) -> TensorDictBase:
        """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
        if batch_size is None:
            batch_size = []
        # Define min and max for observation variables (thetas, thetadots)
        obs_min = torch.concat(
            (-torch.pi*torch.ones(batch_size,self.nb_joints),
             -10*torch.ones(batch_size,self.nb_joints)),
             dim=-1)
        obs_max = torch.concat(
            (torch.pi*torch.ones(batch_size,self.nb_joints),
             10*torch.ones(batch_size,self.nb_joints)),
             dim=-1)

        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "obs_min":obs_min,
                        "obs_max":obs_max,
                        "act_min":obs_min,
                        "act_max":obs_max
                    },
                    [batch_size],
                )
            },
            [batch_size],
        )
        # limb dimension
        lengths = torch.zeros(self.nb_joints, dtype=torch.float32)
        for i in range(self.nb_joints):
            lengths[i] = 1
        self.lengths = lengths

        #if batch_size:
        #    td = td.expand(batch_size).contiguous()
        return td
    
    def _make_spec(self, td_params):
        len_limbs = np.cumsum(self.lengths)

        # observation: theta, theta_dot, limb position
        self.observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                low=td_params["params", "obs_min"],
                high=td_params["params", "obs_max"],
                shape=(batch_size, 2*self.nb_joints,),
                dtype=torch.float32,
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(td_params["params"]),
            shape=td_params.shape,
        )

        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()

        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        #self.action_spec =  CompositeSpec(
        #    action=BoundedTensorSpec(
        #        low=td_params["params", "act_min"],
        #        high=td_params["params", "act_max"],
        #        shape=(2*self.nb_joints),# (thetas, thetadots)
        #        dtype=torch.float32,
        #    ),
        #    shape=td_params.shape,
        #)
        self.action_spec = BoundedTensorSpec(
            low=td_params["params", "act_min"],
            high=td_params["params", "act_max"],
            shape=(batch_size, 2*self.nb_joints),# (thetas, thetadots)
            dtype=torch.float32,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))

    def _step(self, tensordict):
        # current state
        observed_state = tensordict["observation"]
        th, thdot = observed_state[...,:self.nb_joints], observed_state[...,self.nb_joints:]

        # action
        action = tensordict["action"]
        action = action.clamp(tensordict["params", "act_min"],
                              tensordict["params", "act_max"])
        th_a, thdot_a = action[...,:self.nb_joints], action[...,self.nb_joints:]
        
        new_th = pd_controller(th_a-th, thdot_a-thdot)
        new_thdot = (new_th-th)/self.dt
        new_thdot = new_thdot.clamp(tensordict["params", "act_min"][...,self.nb_joints:],
                                    tensordict["params", "act_max"][...,self.nb_joints:])
        new_th = torch.remainder(th + new_thdot * self.dt, 2*torch.pi)-torch.pi
        
        new_obs = torch.concat((new_th, new_thdot), dim=-1)

        new_pos = update_pos(new_th, self.lengths)

        reward = imitation_reward(new_th, new_thdot, th_a, thdot_a)
        reward = reward.view(*tensordict.shape, 1)
        #reward = reward.view(-1, 1)
        done = torch.zeros_like(reward, dtype=torch.bool)

        out = TensorDict(
            {
                "observation": new_obs,
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
            #torch.Size([reward.shape[0]])
        )
        return out

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=batch_size)

        obs_max = tensordict["params", "obs_max"]
        obs_min = tensordict["params", "obs_min"]

        size = (batch_size, 2*self.nb_joints)

        # for non batch-locked environments, the input ``tensordict`` shape dictates the number
        # of simulators run simultaneously. In other contexts, the initial
        # random state's shape will depend upon the environment batch-size instead.
        obs = (
            torch.rand(size, generator=self.rng, device=self.device)
            * (obs_max - obs_min)
            + obs_min
        )
        
        pos = update_pos(obs[...,:self.nb_joints], self.lengths)

        out = TensorDict(
            {
                "observation": obs,
                "params": tensordict["params"],
            },
            batch_size=batch_size,
            device="cpu",
        )
        return out
        
    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng


if __name__ == '__main__':
    env = FlyEnv()
    check_env_specs(env)

    #rollout = env.rollout(
    #    3,
    #    auto_reset=False,  # we're executing the reset out of the ``rollout`` call
    #    tensordict=env.reset(env.gen_params(batch_size=[10])),
    #)
    #print(rollout)