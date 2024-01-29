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
"""Implementation of the Policy Gradient algorithm."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed
from utils.config import load_config, DotDict
from utils.replay_buffer import ReplayBuffer


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods,line-too-long
class PolicyGradient(BaseAlgo):
    """The Policy Gradient algorithm.

    References:
        - Title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        - Authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.
        - URL: `PG <https://proceedings.neurips.cc/paper/1999/file64d828b85b0bed98e80ade0a5c43b0f-Paper.pdf>`_
    """

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: OnPolicyAdapter = OnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        # assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
        #     distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        # ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        # self._steps_per_epoch: int = (
        #     self._cfgs.algo_cfgs.steps_per_epoch
        #     // distributed.world_size()
        #     // self._cfgs.train_cfgs.vector_env_nums
        # )
        self._steps_per_epoch = self._cfgs.algo_cfgs.steps_per_epoch

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)
        
        if not self._cfgs.train_cfgs.train:
            for param in self._actor_critic.actor.parameters():
                param.requires_grad = False
            for param in self._actor_critic.reward_critic.parameters():
                param.requires_grad = False
            if self._cfgs.algo_cfgs.use_cost:
                for param in self._actor_critic.cost_critic.parameters():
                    param.requires_grad = False

        # if distributed.world_size() > 1:
        #     distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        # self._buf: VectorOnPolicyBuffer = VectorOnPolicyBuffer(
        #     obs_space=self._env.observation_space,
        #     act_space=self._env.action_space,
        #     size=self._steps_per_epoch,
        #     gamma=self._cfgs.algo_cfgs.gamma,
        #     lam=self._cfgs.algo_cfgs.lam,
        #     lam_c=self._cfgs.algo_cfgs.lam_c,
        #     advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
        #     standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
        #     standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
        #     penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
        #     num_envs=self._cfgs.train_cfgs.vector_env_nums,
        #     device=self._device,
        # )
        current_state = self._env._env.__getattr__('_env')._env.get_state()
        net_input = self._env._env.__getattr__('_env')._obs_wrapper.query(current_state)
        net_output = self._actor_critic.sample_action(net_input)
        self._buf = ReplayBuffer(self._env._env._config, net_input, net_output)

    def _init_log(self) -> None:
        """Log info about epoch.

        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost        | Average cost of the epoch.                                           |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet         | Average return of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Values/reward         | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-----------------------+----------------------------------------------------------------------+
        | Values/cost           | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-----------------------+----------------------------------------------------------------------+
        | Values/Adv            | Average reward advantage of the epoch.                               |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi          | Loss of the policy network.                                          |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic | Loss of the cost critic network.                                     |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy         | Entropy of the policy network.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Train/StopIters       | Number of iterations of the policy network.                          |
        +-----------------------+----------------------------------------------------------------------+
        | Train/PolicyRatio     | Ratio of the policy network.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Train/LR              | Learning rate of the policy network.                                 |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key('Metrics/EpRet', window_length=50)
        self._logger.register_key('Metrics/EpCost', window_length=50)
        self._logger.register_key('Metrics/EpLen', window_length=50)
        self._logger.register_key('Metrics/Succ', window_length=50)
        
        self._logger.register_key('Rewards/obj_dis_reward', window_length=50)
        self._logger.register_key('Rewards/reach_reward', window_length=50)
        self._logger.register_key('Rewards/action_pen', window_length=50)
        self._logger.register_key('Rewards/contact_reward', window_length=50)
        self._logger.register_key('Rewards/lift_reward', window_length=50)
        self._logger.register_key('Rewards/real_obj_height', window_length=50)
        self._logger.register_key('Rewards/tpen', window_length=50)
        self._logger.register_key('Rewards/reward', window_length=50)
        self._logger.register_key('Rewards/cost', window_length=50)

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio', min_and_max=True)
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type in ['gaussian_learning', 'pointnet_mlp_actor']:
            self._logger.register_key('Train/PolicyStd')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')
        
        self._step_size = self._cfgs.model_cfgs.actor.lr
        
        args = self._env._env._config

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()

            with torch.no_grad():
                rollout_time = time.time()
                self._env.rollout(
                    steps_per_epoch=self._steps_per_epoch,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                )
                self._logger.store({'Time/Rollout': time.time() - rollout_time})
            
            print('available', self._buf.storage['available'][self._buf.step:self._buf.step+self._buf.inner_iters, 0])
            print('reward', self._buf.storage['reward'][self._buf.step:self._buf.step+self._buf.inner_iters, 0])
            print('value_r', self._buf.storage['value'][self._buf.step:self._buf.step+self._buf.inner_iters, 0])
            print('return', self._buf.storage['return'][self._buf.step:self._buf.step+self._buf.inner_iters, 0])
            print('advantage', self._buf.storage['advantage'][self._buf.step:self._buf.step+self._buf.inner_iters, 0])
            print('cost', self._buf.storage['cost'][self._buf.step:self._buf.step+self._buf.inner_iters, 0])
            print('value_c', self._buf.storage['value_c'][self._buf.step:self._buf.step+self._buf.inner_iters, 0])
            print('return_c', self._buf.storage['return_c'][self._buf.step:self._buf.step+self._buf.inner_iters, 0])
            print('advantage_c', self._buf.storage['advantage_c'][self._buf.step:self._buf.step+self._buf.inner_iters, 0])
            
            if epoch < (args.init_timesteps + args.act_timesteps) / args.inner_iters + 1:
                continue

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            # if self._cfgs.model_cfgs.actor.lr is not None:
            #     self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._step_size,
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. hint::

            +----------------+------------------------------------------------------------------+
            | obs            | ``observation`` sampled from buffer.                             |
            +================+==================================================================+
            | act            | ``action`` sampled from buffer.                                  |
            +----------------+------------------------------------------------------------------+
            | target_value_r | ``target reward value`` sampled from buffer.                     |
            +----------------+------------------------------------------------------------------+
            | target_value_c | ``target cost value`` sampled from buffer.                       |
            +----------------+------------------------------------------------------------------+
            | logp           | ``log probability`` sampled from buffer.                         |
            +----------------+------------------------------------------------------------------+
            | adv_r          | ``estimated advantage`` (e.g. **GAE**) sampled from buffer.      |
            +----------------+------------------------------------------------------------------+
            | adv_c          | ``estimated cost advantage`` (e.g. **GAE**) sampled from buffer. |
            +----------------+------------------------------------------------------------------+


        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        #. Repeat steps 2, 3, 4 until the KL divergence violates the limit.
        """
        
        args = self._env._env._config
        
        update_counts = 0
        final_kl = 0.0
        
        for epoch in range(args.ppo_epochs):
            for batch in self._buf.get_batches(args.batch_size):
                net_output = self._actor_critic.evaluate({k: batch[k] for k in ['robot_state_stacked', 'visual_observation', 'progress_buf', 'goal']}, batch['raw_action'])
                
                kl = torch.sum(net_output['sigma'].log() - batch['sigma'].log() + (torch.square(batch['sigma']) + torch.square(batch['mu'] - net_output['mu'])) / (2.0 * torch.square(net_output['sigma'])) - 0.5, dim=-1)
                kl_mean = kl.mean()
                
                # if kl_mean > args.desired_kl * 2.0:
                #     self._step_size = max(1e-5, self._step_size / 1.5)
                # elif kl_mean < args.desired_kl / 2.0 and kl_mean > 0.0:
                #     self._step_size = min(1e-2, self._step_size * 1.5)
                
                for param_group in self._actor_critic.actor_critic_optimizer.param_groups:
                    param_group['lr'] = self._step_size
                
                # if kl_mean > args.desired_kl:
                #     break
                
                value_loss = (batch['return'] - net_output['value']).square().mean()
                if self._cfgs.algo_cfgs.use_cost:
                    cost_loss = (batch['return_c'] - net_output['value_c']).square().mean()
                        
                ratio = torch.exp(net_output['log_prob'] - batch['log_prob'])
                # print('ratio', ratio.min(), ratio.max(), ratio.std(), ratio.mean())
                advantage = self._compute_adv_surrogate(batch['advantage'], batch['advantage_c'])
                surrogate = -torch.squeeze(advantage) * ratio
                surrogate_clipped = -torch.squeeze(advantage) * torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                loss = surrogate_loss + args.value_loss_weight * value_loss + args.cost_loss_weight * (cost_loss if self._cfgs.algo_cfgs.use_cost else 0)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._actor_critic.parameters(), args.max_grad_norm)
                self._actor_critic.actor_critic_optimizer.step()
                self._actor_critic.actor_critic_optimizer.zero_grad()
                
                self._logger.store({'Loss/Loss_reward_critic': value_loss.mean().item()})
                if self._cfgs.algo_cfgs.use_cost:
                    self._logger.store({'Loss/Loss_cost_critic': cost_loss.mean().item()})
                self._logger.store(
                    {
                        'Train/Entropy': 0,
                        'Train/PolicyRatio': ratio,
                        'Train/PolicyStd': self._actor_critic.actor.policy.log_std.exp().mean().item(),
                        'Loss/Loss_pi': surrogate_loss.mean().item(),
                    },
                )
                
            
            final_kl = kl_mean.item()
            update_counts += 1

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': advantage.mean().item(),
                'Train/KL': final_kl,
            },
        )

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_r (torch.Tensor): The ``target_value_r`` sampled from buffer.
        """
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()

        self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        # self._actor_critic.actor_optimizer.step()
    
    def _update_actor_critic(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        target_value_r: torch.Tensor,
        target_value_c: torch.Tensor,
    ) -> None:
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss_pi = self._loss_pi(obs, act, logp, adv)
        obs_feature = self._actor_critic.actor.get_obs_feature(obs)
        predicted_value_r = self._actor_critic.reward_critic(obs_feature)
        loss_v = nn.functional.mse_loss(predicted_value_r, target_value_r)
        loss = loss_pi + 2 * loss_v
        assert not torch.isnan(loss).any(), 'loss is nan'
        if self._cfgs.train_cfgs.train:
            self._actor_critic.actor_critic_optimizer.zero_grad()
            loss.backward()
            if self._cfgs.algo_cfgs.use_max_grad_norm:
                clip_grad_norm_(
                    self._actor_critic.parameters(),
                    self._cfgs.algo_cfgs.max_grad_norm,
                )
            # distributed.avg_grads(self._actor_critic)
            self._actor_critic.actor_critic_optimizer.step()
        
        self._logger.store({'Loss/Loss_reward_critic': loss_v.mean().item()})
        

    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function of reward to update policy network.
        """
        return adv_r

    def _loss_pi(
        self,
        batch
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Policy Gradient, the loss is defined as:

        .. math::

            L = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} [
                \sum_{t=0}^T ( \frac{\pi^{'}_{\theta}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} )
                 A^{R}_{\pi_{\theta}}(s_t, a_t)
            ]

        where :math:`\pi_{\theta}` is the policy network, :math:`\pi^{'}_{\theta}`
        is the new policy network, :math:`A^{R}_{\pi_{\theta}}(s_t, a_t)` is the advantage.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        
        net_output = self._actor_critic.evaluate({k: batch[k] for k in ['robot_state_stacked', 'visual_observation', 'progress_buf', 'goal']}, batch['raw_action'])
        ratio = torch.exp(net_output['log_prob'] - batch['log_prob'])
        loss = -(ratio * torch.squeeze(batch['advantage'])).mean()
        self._logger.store(
            {
                'Train/Entropy': 0,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': self._actor_critic.actor.policy.log_std.exp().mean().item(),
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        
        return loss
