"""
Last modified date: 2023.07.01
Author: Jialiang Zhang
Description: policy network
"""

import math
import numpy as np
import torch
import torch.nn as nn
from nflows.flows import ConditionalGlow
from einops import rearrange
from torch.distributions import Distribution, Normal

from utils.robot_info import UR5_PARAMS
from networks.mlp import MLP

class Policy(nn.Module):
    def __init__(
        self, 
        output_type='raw_delta',
        normalize_output=True,
        **kwargs
    ):
        super(Policy, self).__init__()
        self.output_type = output_type
        self.register_buffer('arm_param', UR5_PARAMS)
        self.normalize_output = normalize_output
        self.register_buffer('output_mean', torch.zeros(22))
        self.register_buffer('output_std', torch.ones(22))
        self.scale = kwargs.get('scale', 1)
        self.inv = kwargs.get('inv', 0)
        self.strict_range = kwargs.get('strict_range', 0)
    
    def forward(self, robot_state_stacked, visual_observation, output_raw=False):
        return self.pred_action(robot_state_stacked, visual_observation, output_raw)
    
    def calculate_loss(self, robot_state, observation_feature, gt_actions, with_differentiable_action):
        raise NotImplementedError('This is an abstract method')
    
    def log_prob(self, robot_state_stacked, visual_observation, gt_actions):
        raise NotImplementedError('This is an abstract method')
    
    def sample_action(self, robot_state, action, output_raw=False):
        raise NotImplementedError('This is an abstract method')

    def pred_action(self, robot_state, action, output_raw=False):
        # expect the input to be raw output of the policy network
        if not output_raw:
            action = self.raw_action_to_action(robot_state, action)
        return action
    
    def get_normalize_parameters(self, loader):
        if self.normalize_output:
            assert self.scale == 1
            assert not self.inv
            raw_actions = []
            i = 0
            for batch in loader:
                if i > 1000:
                    break
                robot_state = batch['robot_state_stacked'][:, 0]
                i += 1
                action = batch['action'][:, :22]
                raw_action = self.action_to_raw_action(robot_state, action)
                raw_actions.append(raw_action)
            raw_actions = torch.cat(raw_actions, dim=0)
            self.output_mean[:] = raw_actions.mean(dim=0)
            self.output_std[:] = raw_actions.std()
            

    def raw_action_to_action(self, robot_state, raw_action):
        if self.strict_range:
            raw_action = torch.tanh(raw_action)
            raw_action = torch.clamp(raw_action, -0.95, 0.95)
        raw_action = raw_action.reshape(raw_action.shape[0], -1, 22)
        raw_action = raw_action * self.scale * self.output_std + self.output_mean
        if self.output_type == 'raw':
            result = raw_action
        elif self.output_type == 'raw_delta':
            action = robot_state[:, None, :22] + raw_action
            result = self.regularize_angles(action)
        elif self.output_type == 'raw_action_delta':
            raise NotImplementedError()
            prev_action = robot_state[:, -44:-22]
            action = prev_action + raw_action
            return self.regularize_angles(action)
        elif self.output_type == 'ee_delta':
            raise NotImplementedError()
            ee_dpos = raw_action[:, :6]
            jacobian = compute_arm_jacobian(self.arm_param, robot_state[:, :6])
            damping = 0.05
            # solve damped least squares
            jacobian_T = torch.transpose(jacobian, -1, -2)
            lmbda = torch.eye(6, device=jacobian.device) * (damping ** 2)
            arm_dpos = torch.einsum('nab,nbc,nc->na', jacobian_T, torch.inverse(torch.bmm(jacobian, jacobian_T) + lmbda), ee_dpos)
            hand_dpos = raw_action[:, 6:]
            dpos = torch.cat([arm_dpos, hand_dpos], dim=-1)
            action = robot_state[:, :22] + dpos
            result = self.regularize_angles(action)
        else:
            raise NotImplementedError()
        return result.reshape(result.shape[0], -1)

    def action_to_raw_action(self, robot_state, action):
        action = action.reshape(action.shape[0], -1, 22)
        if self.output_type == 'raw':
            result = action
        elif self.output_type == 'raw_delta':
            result = action - robot_state[:, None, :22]
        elif self.output_type == 'raw_action_delta':
            raise NotImplementedError()
            prev_action = robot_state[:, -44:-22]
            delta_action = action - prev_action
        elif self.output_type == 'ee_delta':
            raise NotImplementedError()
            dpos = action - robot_state[:, :22]
            arm_dpos = dpos[:, :6]
            hand_dpos = dpos[:, 6:]
            jacobian = compute_arm_jacobian(self.arm_param, robot_state[:, :6])
            damping = 0.05
            # solve damped least squares
            jacobian_T = torch.transpose(jacobian, -1, -2)
            lmbda = torch.eye(6, device=jacobian.device) * (damping ** 2)
            ee_dpos = torch.einsum('nab,nb->na', torch.inverse(torch.bmm(jacobian_T, torch.inverse(torch.bmm(jacobian, jacobian_T) + lmbda))), arm_dpos)
            result = torch.cat([ee_dpos, hand_dpos], dim=-1)
        else:
            raise NotImplementedError()
        result = ((result - self.output_mean) / self.output_std / self.scale).reshape(result.shape[0], -1)
        if self.strict_range:
            result = torch.clamp(result, -0.95, 0.95)
            result = torch.atanh(result)
        return self.regularize_angles(result)
    
    def regularize_angles(self, action):
        return (action + np.pi) % (2 * np.pi) - np.pi


class GaussianPolicy(Policy):
    
    _current_dist: Normal
    
    def __init__(self, observation_dim, **kwargs):
        super().__init__(**kwargs)
        policy_mlp_parameters=dict(
            hidden_layers_dim=[1024, 1024, 512, 512], 
            output_dim=22 * (2 if self.inv else 1), 
            act=kwargs.get('act_fn', None)
        )
        self.policy = MLP(input_dim=observation_dim, **policy_mlp_parameters)
        if kwargs['with_std']:
            self.log_std = nn.Parameter(torch.full((22,), np.log(kwargs['init_std'])))
        else:
            self.register_buffer('log_std', torch.zeros((22,)))
    
    def calculate_loss(self, robot_state, observation_feature, gt_actions, with_differentiable_action):
        pred_raw_actions = self.pred_action(robot_state, observation_feature, output_raw=True)
        gt_raw_actions = self.action_to_raw_action(robot_state, gt_actions)
        loss = (pred_raw_actions - gt_raw_actions).square().mean(dim=-1)
        if not with_differentiable_action:
            return loss
        else:
            pred_actions = self.raw_action_to_action(robot_state, pred_raw_actions)
            return loss, pred_actions
    
    def _log_prob(self, mu, sigma, x):
        assert not self.inv
        return -((mu - x).square() / (sigma ** 2)).sum(dim=-1) / 2 - np.log(np.sqrt(2 * np.pi)) * mu.shape[-1] - sigma.log().sum()
    
    def log_prob(self, robot_state, observation_feature, gt_actions, input_raw=False):
        assert not self.inv
        pred_raw_actions = self.pred_action(robot_state, observation_feature, output_raw=True)
        gt_raw_actions = self.action_to_raw_action(robot_state, gt_actions) if not input_raw else gt_actions
        return self._log_prob(pred_raw_actions, self.log_std.exp(), gt_raw_actions)

    def pred_action(self, robot_state, observation_feature, output_raw=False):
        actions = self.policy(observation_feature)
        return super().pred_action(robot_state, actions, output_raw)
    
    def sample_action(self, robot_state, observation_feature, output_raw=False):
        assert not self.inv
        mu = self.policy(observation_feature)
        actions = mu + torch.randn_like(mu) * self.log_std.exp()

        return dict(action=super().pred_action(robot_state, actions, output_raw),
                    raw_action=actions,
                    log_prob=self._log_prob(mu, self.log_std.exp(), actions),
                    mu=mu,
                    sigma=self.log_std.exp().expand(len(mu), -1))

    def _distribution(self, obs: torch.Tensor) -> Normal:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The normal distribution of the mean and standard deviation from the actor.
        """
        mean = self.policy(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

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
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        if deterministic:
            return self._current_dist.mean
        return self._current_dist.rsample()

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
        return self._current_dist.log_prob(act).sum(axis=-1)

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return torch.exp(self.log_std).mean().item()

    @std.setter
    def std(self, std: float) -> None:
        device = self.log_std.device
        self.log_std.data.fill_(torch.log(torch.tensor(std, device=device)))

