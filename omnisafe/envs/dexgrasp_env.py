import sys

sys.path.append('/home/jialiangzhang/Workspace/omnisafe/omnisafe/envs/')

from gymnasium import spaces
from simulation.isaac import Env
from typing import Any, ClassVar
import numpy as np
import torch
from omnisafe.envs.core import CMDP, env_register
from utils.obs_wrapper import ObsWrapper
from utils.config import load_config, DotDict


@env_register
class SafetyDexgraspEnv(CMDP):
    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False
    _support_envs = [
        'dexgrasp-v0',
    ]
    
    def __init__(
        self, 
        env_id,
        num_envs,
        device,
        **kwargs,
    ):
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)
        
        # load config
        self._config = load_config('config/isaac.yaml', DotDict())
        self._config = load_config('config/rl.yaml', self._config)
        
        # obs wrapper
        self._obs_wrapper = ObsWrapper(self._config)
        
        # initialize env
        self._env = Env(self._config)
        self._env.reset()
        self._env.randomize_start()
        
        # set parameters
        obs_dim = 409
        act_dim = 22
        self._observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype='float32')
        self._action_space = spaces.Box(low=-1, high=1, shape=(act_dim,), dtype='float32')
        self._metadata = dict()
    
    def step(
        self,
        action,
    ):
        self._env.step(action)
        
        terminated = self._env.progress_buf == self._config.act_timesteps
        truncated = torch.zeros(self._num_envs, dtype=torch.bool, device=action.device)
        reset_indices = (self._env.progress_buf == self._config.act_timesteps).nonzero().reshape(-1)
        self._env._reset(indices=reset_indices)
        
        current_state = self._env.get_state()
        
        self._obs_wrapper.reset(current_state, indices=reset_indices)
        net_input = self._obs_wrapper.query(current_state)
        
        # TODO: calculate cost
        cost = current_state['cost']
        
        info = dict(
            success=self._env.record_success, 
            obj_dis_reward=current_state['obj_dis_reward'],
            reach_reward=current_state['reach_reward'],
            action_pen=current_state['action_pen'],
            contact_reward=current_state['contact_reward'],
            lift_reward=current_state['lift_reward'],
            real_obj_height=current_state['real_obj_height'],
            tpen=current_state['tpen'],
            reward=current_state['reward'], 
        )
        
        return net_input, current_state['reward'], cost, terminated, truncated, info
    
    def reset(
        self,
        seed,
        options=None,
    ):
        self._env.reset()
        current_state = self._env.get_state()
        net_input = self._obs_wrapper.query(current_state)
        return net_input, dict()
    
    def set_seed(self, seed):
        self.reset(seed=seed)
    
    def sample_action(self):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError
    
    def close(self):
        self._env.close()


if __name__ == '__main__':
    env = SafetyDexgraspEnv('dexgrasp-v0', 1, 'cpu')
    env.reset(0, None)
    env.close()
