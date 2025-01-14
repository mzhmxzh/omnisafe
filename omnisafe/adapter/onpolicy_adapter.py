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
"""OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from rich.progress import track

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config


class OnPolicyAdapter(OnlineAdapter):
    """OnPolicy Adapter for OmniSafe.

    :class:`OnPolicyAdapter` is used to adapt the environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._reset_log()

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        # self._reset_log()

        # obs, info = self.reset()
        current_state = self._env.__getattr__('_env')._env.get_state()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            net_input = self._env.__getattr__('_env')._obs_wrapper.query(current_state)
            net_output = agent.sample_action(net_input)

            _, _, _, terminated, truncated, _ = self.step(net_output['action'])
            current_state = self._env.__getattr__('_env')._env.get_state()
            
            buffer.update(net_input, net_output, current_state['reward'], current_state['cost'])

            self._log_value(reward=current_state['reward'], cost=current_state['cost'], info=current_state)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': net_output['value_c'].clone()})
            logger.store({'Value/reward': net_output['value'].clone()})
            
            for key in ['obj_dis_reward', 'reach_reward', 'action_pen', 'contact_reward', 'lift_reward', 'real_obj_height', 'tpen', 'reward', 'cost']:
                logger.store({'Rewards/' + key: current_state[key].clone()})

            # # sanity check
            # agent.actor(obs)
            # new_logp = agent.actor.log_prob(act)
            # assert torch.allclose(logp, new_logp), 'logp is not equal to new_logp'
            
            
            # # sanity check
            # for j in range(step + 1):
            #     buffer_step = (buffer.step + buffer.inner_iters * 2 - 1 - j) % (buffer.inner_iters * 2)
            #     agent.actor(buffer.storage['net_input'][buffer_step])
            #     new_logp = agent.actor.log_prob(buffer.storage['action'][buffer_step])
            #     assert torch.allclose(buffer.storage['logp'][buffer_step], new_logp), 'logp is not equal to new_logp'
            # buffer_step = (buffer.step + buffer.inner_iters * 2 - 1 - step) % (buffer.inner_iters * 2)
            # agent.actor(buffer.storage['net_input'][buffer_step:buffer_step + step + 1].reshape(-1, *buffer.storage['net_input'].shape[2:]))
            # new_logp = agent.actor.log_prob(buffer.storage['action'][buffer_step:buffer_step + step + 1].reshape(-1, *buffer.storage['action'].shape[2:]))
            # if not torch.allclose(buffer.storage['logp'][buffer_step:buffer_step + step + 1].reshape(-1, *buffer.storage['logp'].shape[2:]), new_logp):
            #     print('logp is not equal to new_logp')

            # obs = next_obs
            # epoch_end = step >= steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if done or time_out:

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        logger.store(
                            {
                                'Metrics/Succ': self._env.__getattr__('_env')._env.record_success[idx].float().clone(), 
                            }
                        )
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
        
        net_input = self._env.__getattr__('_env')._obs_wrapper.query(current_state)
        net_output = agent.sample_action(net_input)
        buffer.compute_returns(net_output['value'], net_output['value_c'], self._cfgs.algo_cfgs.gamma, self._cfgs.algo_cfgs.lam)

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx].clone(),
                'Metrics/EpCost': self._ep_cost[idx].clone(),
                'Metrics/EpLen': self._ep_len[idx].clone(),
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
