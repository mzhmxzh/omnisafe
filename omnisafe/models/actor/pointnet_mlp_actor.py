# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of PointnetMLPActor."""

from __future__ import annotations
import sys

sys.path.append('/home/liuhaoran/code/omnisafe/omnisafe/envs/')
sys.path.append('/home/jialiangzhang/Workspace/omnisafe/omnisafe/envs/')
sys.path.append('/mnt/disk0/danshili/Workspace/omnisafe/omnisafe/envs/')

import numpy as np
import os
import json
from gymnasium import spaces
import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

from omnisafe.models.actor.gaussian_actor import GaussianActor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network

from utils.config import load_config, DotDict
from networks.backbone import FeatureExtractor
from networks.policy import GaussianPolicy

# pylint: disable-next=too-many-instance-attributes
class PointnetMLPActor(GaussianActor):
    """Implementation of PointnetMLPActor.

    GaussianLearningActor is a Gaussian actor with a learnable standard deviation. It is used in
    on-policy algorithms such as ``PPO``, ``TRPO`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    _current_dist: Normal

    def __init__(self):
        """Initialize an instance of :class:`PointnetMLPActor`."""
        # load config
        self._config = DotDict(
            dict(
                split_path='data/splits-v15/bc_dataset.json',
                specify_obj='sem-Car-da496ba5f90a476a1493b1a3f79fe4c6_006',
                simulate_mode='train',
                obj_num=None,
            )
        )
        if self._config.specify_obj is not None:
            self._config.object_code = [self._config.specify_obj]
        else:
            with open(self._config.split_path, 'r') as json_file:
                self._config.object_code = json.load(json_file)[self._config.simulate_mode]
                self._config.object_code = [code for code in self._config.object_code if not 'ycb' in code and len(np.load(os.path.join('data', 'results_filtered-v15', code + '.npy'), allow_pickle=True))]
            if self._config.obj_num is not None:
                self._config.object_code = self._config.object_code[:self._config.obj_num]
        self._config = load_config('config/rl.yaml', self._config)
        self._config = load_config('config/isaac.yaml', self._config)
        self._config = self._config.actor
        obs_dim = 409
        act_dim = 22
        hidden_sizes = [1024, 1024, 512, 512]
        activation = 'tanh'
        weight_initialization_mode = 'orthogonal'
        obs_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype='float32')
        act_space = spaces.Box(low=-1, high=1, shape=(act_dim,), dtype='float32')
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        
        # initialize actor
        self.feature_extractor = FeatureExtractor(self._config)
        
        self.observation_dim = self.feature_extractor.observation_dim
        self.observation_dim *= self._config.num_frames
        
        self.policy = GaussianPolicy(self.observation_dim, **self._config.policy_parameters)

        # self.mean: nn.Module = build_mlp_network(
        #     sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim],
        #     activation=activation,
        #     weight_initialization_mode=weight_initialization_mode,
        # )
        # self.log_std: nn.Parameter = nn.Parameter(torch.zeros(self._act_dim), requires_grad=True)
        def init_weights(sequential, scales):
                [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
                enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        robot_mlp_weights = [np.sqrt(2)] * 3
        init_weights(self.feature_extractor.robot_mlp.mlp, robot_mlp_weights)

        actor_weights = [np.sqrt(2)] * 4
        actor_weights.append(0.01)
        init_weights(self.policy.policy.mlp, actor_weights)
    
    def train_parameters(self):
        return list(self.policy.parameters()) + list(self.feature_extractor.parameters())

    def pred_action(self, robot_state, observation_feature, output_raw=False, goal=None):
        if goal is not None:
            observation_feature = torch.cat([observation_feature, goal], dim=1)
        return self.policy.pred_action(robot_state, observation_feature, output_raw)

    def sample_action(self, robot_state, observation_feature, output_raw=False, goal=None):
        if goal is not None:
            observation_feature = torch.cat([observation_feature, goal], dim=1)
        return self.policy.sample_action(robot_state, observation_feature, output_raw)
    
    def get_observation_feature(self, robot_state_stacked, visual_observation):
        return self.feature_extractor(robot_state_stacked, visual_observation)
    
    def log_prob2(self, robot_state, observation_feature, raw_action):
        return self.policy.log_prob(robot_state, observation_feature, raw_action, input_raw=True)

    def get_obs_feature(self, obs):
        robot_state_stacked = obs[:, :25].reshape(len(obs), 1, 25)
        visual_observation = obs[:, 25:].reshape(len(obs), -1, 3)
        obs_feature, _ = self.feature_extractor(robot_state_stacked, visual_observation)
        self._robot_state = obs[:, :22].clone()
        self._obs_feature = obs_feature
        return obs_feature

    def _distribution(self, obs: torch.Tensor):
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The normal distribution of the mean and standard deviation from the actor.
        """
        obs_feature = self.get_obs_feature(obs)
        return self.policy(obs_feature)

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action given observation.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the mean of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            The mean of the distribution if deterministic is True, otherwise the sampled action.
        """
        squeeze = False
        if len(obs.shape) == 1:
            squeeze = True
            obs = obs.unsqueeze(0)
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        if deterministic:
            return self._current_dist.mean
        raw_action = self._current_dist.rsample()
        action = self.policy.raw_action_to_action(self._robot_state, raw_action)
        if squeeze:
            action = action.squeeze(0)
        return action
        return raw_action

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward method.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The current distribution.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action given the current distribution.

        .. warning::
            You must call :meth:`forward` or :meth:`predict` before calling this method.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .

        Returns:
            Log probability of the action.
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        if len(act.shape) == 1:
            raw_action = self.policy.action_to_raw_action(self._robot_state, act.unsqueeze(0)).squeeze(0)
        else:
            raw_action = self.policy.action_to_raw_action(self._robot_state, act)
        return self._current_dist.log_prob(raw_action).sum(axis=-1)
        return self._current_dist.log_prob(act).sum(axis=-1)

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return torch.exp(self.policy.log_std).mean().item()

    @std.setter
    def std(self, std: float) -> None:
        device = self.policy.log_std.device
        self.policy.log_std.data.fill_(torch.log(torch.tensor(std, device=device)))


if __name__ == '__main__':
    actor = PointnetMLPActor()
    print(actor)
