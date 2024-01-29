import torch
import numpy as np
from collections import deque

class ReplayBuffer():
    def __init__(self, args, input_dict, output_dict, double=True, with_next_state=False, gpu2=False):
        self.config = args
        self.inner_iters = args.inner_iters
        self.device = args.gpu2 if gpu2 else args.gpu
        self.double = double
        self.scale = 2 if double else 1

        self.storage = dict()
        for example in [input_dict, output_dict]:
            for k, v in example.items():
                if type(v) == torch.Tensor:
                    self.storage[k] = torch.zeros_like(v)[None].repeat_interleave(self.inner_iters * self.scale, dim=0)
                    self.num_envs = len(v)
                else:
                    self.storage[k] = v
        if with_next_state:
            for k, v in input_dict.items():
                if type(v) == torch.Tensor:
                    self.storage['next_' + k] = torch.zeros_like(v)[None].repeat_interleave(self.inner_iters * self.scale, dim=0)
                else:
                    self.storage['next_' + k] = v
        self.storage['return'] = torch.zeros((self.inner_iters * self.scale, self.num_envs), device=self.device)
        self.storage['advantage'] = torch.zeros((self.inner_iters * self.scale, self.num_envs), device=self.device)
        self.step = 0
    
    def update(self, current_state, output, reward):
        for data_dict in [current_state, output]:
            for k, v in data_dict.items():
                if type(v) == torch.Tensor:
                    self.storage[k][self.step] = v
        self.storage['reward'][self.step] = reward
        self.step = (self.step + 1) % (self.inner_iters * self.scale)
    
    def update_next_state(self, next_state):
        assert self.scale == 1
        for k, v in next_state.items():
            if type(v) == torch.Tensor:
                self.storage['next_' + k][(self.step - 1) % self.inner_iters] = v
                for i in range(1, self.inner_iters):
                    self.storage['next_' + k][(self.step - 1 - i) % self.inner_iters] = self.storage[k][(self.step - i) % self.inner_iters]
    
    def compute_returns(self, last_values, gamma, lam):
        assert self.scale == 2
        for j in range(self.step, self.step+self.inner_iters):
            advantage = 0
            for i in reversed(range(self.inner_iters)):
                idx = (j + i) % (2 * self.inner_iters)
                if idx == (self.step - 1) % (2 * self.inner_iters):
                    next_values = last_values
                else:
                    next_values = self.storage['value'][(idx+1) % (2*self.inner_iters)]

                delta = self.storage['reward'][idx] + gamma * next_values - self.storage['value'][idx]
                advantage = delta + gamma * lam * advantage
            self.storage['advantage'][j%(2*self.inner_iters)] = advantage 
        self.storage['return'] = self.storage['advantage'] + self.storage['value']

        advantage = self.storage['advantage'][self.step:self.step+self.inner_iters].reshape(-1)[self.storage['available'][self.step:self.step+self.inner_iters].reshape(-1).nonzero().reshape(-1)]
        self.storage['advantage'] = self.storage['advantage'] / (advantage.std() + 1e-8)
    
    def get_batches(self, batch_size, shuffle=True):
        availabe_indices = self.storage['available'][self.step:self.step+self.inner_iters].reshape(-1).nonzero().reshape(-1)
        if shuffle:
            indices = torch.randperm(len(availabe_indices), device=self.device)
        else:
            indices = torch.arange(len(availabe_indices), device=self.device)
        availabe_indices = availabe_indices[indices]
        result = []
        for i in range(0, len(availabe_indices), batch_size):
            if i + batch_size > len(availabe_indices) and i != 0:
                break
            batch_indices = availabe_indices[i:i+batch_size]
            batch_iters = batch_indices // self.num_envs
            batch_envs = batch_indices % self.num_envs
            batch = dict(iters=batch_iters, envs=batch_envs)
            for k, v in self.storage.items():
                if type(v) == torch.Tensor:
                    batch[k] = v[self.step:self.step+self.inner_iters][batch_iters, batch_envs].clone()
                else:
                    batch[k] = v
            result.append(batch)
        return result
    
    def get_all(self, shuffle=True):
        return self.get_batches(self.inner_iters * self.num_envs, shuffle)[0]

class ReplayBufferConsistency(ReplayBuffer):
    def __init__(self, args, input_dict, output_dict, double=True, with_next_state=False):
        super().__init__(args, input_dict, output_dict, double, with_next_state)
        self.buffer_len = args.consistency_buffer_len
        self.buffer_idx = 0
        self.is_buffer_filled = False
        self.buffer_keys = ['obs_feature','action','reward', 'robot_state_stacked', 'visual_observation', 'next_robot_state_stacked','next_visual_observation']
        self.buffer = dict()
        self.buffer['buf_obs_feature'] = torch.zeros_like(output_dict['obs_feature'][:1],device='cpu').repeat_interleave(self.buffer_len, dim=0)
        self.buffer['buf_action'] = torch.zeros_like(output_dict['action'][:1],device='cpu').repeat_interleave(self.buffer_len, dim=0)
        self.buffer['buf_reward'] = torch.zeros_like(input_dict['reward'][:1],device='cpu').repeat_interleave(self.buffer_len, dim=0)
        self.buffer['buf_visual_observation'] = torch.zeros_like(input_dict['visual_observation'][:1],device='cpu').repeat_interleave(self.buffer_len, dim=0)
        self.buffer['buf_robot_state_stacked'] = torch.zeros_like(input_dict['robot_state_stacked'][:1],device='cpu').repeat_interleave(self.buffer_len, dim=0)
        self.buffer['buf_next_robot_state_stacked'] = torch.zeros_like(input_dict['robot_state_stacked'][:1],device='cpu').repeat_interleave(self.buffer_len, dim=0)
        self.buffer['buf_next_visual_observation'] = torch.zeros_like(input_dict['visual_observation'][:1],device='cpu').repeat_interleave(self.buffer_len, dim=0)

    def compute_returns(self, net_output, gamma, lam):
        return
        # assert self.scale == 1
        # for j in range(self.step, self.step+self.inner_iters):
        #     self.storage['min_next_value'][j%(self.inner_iters)] = torch.min(torch.cat([self.storage['next_value_Q1_ema'],self.storage['next_value_Q2_ema']]),0)[0][j%(self.inner_iters)]
        #     bellman_Q1 = self.storage['reward'] + self.config.gamma * self.storage['min_next_value'] - self.storage['value_Q1']
        #     bellman_Q2 = self.storage['reward'] + self.config.gamma * self.storage['min_next_value'] - self.storage['value_Q2']
        #     self.storage['bellman_Q1'][j%(self.inner_iters)] = bellman_Q1[j%(self.inner_iters)]
        #     self.storage['bellman_Q2'][j%(self.inner_iters)] = bellman_Q2[j%(self.inner_iters)]

    def update_next_Q(self, current_state):
        for data_dict in [current_state]:
            for k, v in data_dict.items():
                if k in ['robot_state_stacked','visual_observation']:
                    if type(v) == torch.Tensor:
                        #self.storage['next_' + k][self.step-1] = v
                        buffer_update_indices = ((self.available_indices+self.buffer_idx) % self.buffer_len).long()
                        self.buffer['buf_next_' + k][buffer_update_indices] = v[self.available_indices].detach().cpu()
                        # for i,ind in enumerate(self.available_indices):
                        #     buffer_idx_mod = (self.buffer_idx+i)%self.buffer_len
                        #     self.buffer['buf_next_' + k][buffer_idx_mod] = v[ind].detach().cpu()
                    # else:
                    #     self.storage['next_' + k] = v
        if self.buffer_idx + self.available_indices.shape[0] > self.buffer_len:
            self.is_buffer_filled = True
        self.buffer_idx = (self.buffer_idx + self.available_indices.shape[0]) % self.buffer_len


    def update(self, current_state, output):
        # ['obs_feature','action','reward','next_obs_feature']
        for data_dict in [current_state, output]:
            try:
                self.available_indices = data_dict['available'].nonzero().reshape(-1)
            except:
                pass
            for k, v in data_dict.items():
                if type(v) == torch.Tensor:
                    self.storage[k][self.step] = v
                    if k in self.buffer_keys:
                        buffer_update_indices = ((self.available_indices+self.buffer_idx) % self.buffer_len).long()
                        self.buffer['buf_' + k][buffer_update_indices] = v[self.available_indices].detach().cpu()
                        # for i,ind in enumerate(self.available_indices):
                        #     buffer_idx_mod = (self.buffer_idx+i)%self.buffer_len
                        #     self.buffer['buf_' + k][buffer_idx_mod] = v[ind].detach().cpu()
                    
        self.step = (self.step + 1) % (self.inner_iters * self.scale)

    def draw_sample_from_buffer(self, batch_size,device):
        if self.is_buffer_filled:
            batch_rnd_idx = torch.randint(low=0,high=self.buffer_len,size=(4,int(batch_size/4)))
        else:
            batch_rnd_idx = torch.randint(low=0,high=self.buffer_idx,size=(4,int(batch_size/4)))
        replay_samples = []
        for ind in range(batch_rnd_idx.shape[0]):
            sample = dict()
            for buf in self.buffer_keys:
                sample[buf] = self.buffer['buf_'+buf][batch_rnd_idx[ind]].to(device)
            replay_samples.append(sample)
        return replay_samples

    def inplace_update_next_Q(self, batch, next_Q_values):
        batch['min_next_value'] = torch.min(torch.stack([next_Q_values['next_value_Q1_ema'],next_Q_values['next_value_Q2_ema']]),0)[0]
        bellman_Q1 = batch['reward'] + self.config.gamma * batch['min_next_value'] - batch['value_Q1']
        bellman_Q2 = batch['reward'] + self.config.gamma * batch['min_next_value'] - batch['value_Q2']
        batch['bellman_Q1'] = bellman_Q1
        batch['bellman_Q2'] = bellman_Q2
        return batch
        