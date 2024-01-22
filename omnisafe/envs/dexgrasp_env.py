import sys

sys.path.append('/home/jialiangzhang/Workspace/omnisafe/omnisafe/envs/')

from simulation.isaac import Env
import numpy as np
import torch
from omnisafe.envs.core import CMDP, env_register
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
        config = load_config('config/isaac.yaml', DotDict())
        config = load_config('config/rl.yaml', config)
        
        # initialize env
        self._env = Env(config)
        self._env.reset()
        self._env.randomize_start()
    
    def step(
        self,
        action,
    ):
        current_state = self._env.step(action)
        
        
    def reset(
        self,
        seed,
        options,
    ):
        pass
    
    def set_seed(self, seed):
        self.reset(seed=seed)
    
    def sample_action(self):
        pass
    
    def render(self):
        pass
    
    def close(self):
        pass
    