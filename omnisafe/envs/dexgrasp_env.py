import sys

sys.path.append('/home/jialiangzhang/Workspace/omnisafe/omnisafe/envs/')

from simulation.isaac import Env
import numpy as np
import torch
from omnisafe.envs.core import CMDP, env_register
from utils.obs_wrapper import ObsWrapper
from utils.config import load_config, DotDict


@env_register
class SafetyDexgraspEnv(CMDP):
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
        
        return net_input, current_state['reward'], cost, terminated, truncated, None
    
    def reset(
        self,
        seed,
        options,
    ):
        self._env.reset()
        current_state = self._env.get_state()
        net_input = self._obs_wrapper.query(current_state)
        return net_input, None
    
    def set_seed(self, seed):
        self.reset(seed=seed)
    
    def sample_action(self):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError
    
    def close(self):
        self._env.close()
