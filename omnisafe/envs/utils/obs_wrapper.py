import torch
try:
    import MinkowskiEngine as ME
except:
    pass

class ObsWrapper():
    def __init__(self, config, use_camera=None):
        self.config = config
        # self.last_qpos = torch.zeros((config.num_envs, 22), device=config.gpu)
        # self.last_vel = torch.zeros((config.num_envs, 22), device=config.gpu)
        # self.last_qpos_action = torch.zeros((config.num_envs, 22), device=config.gpu)
        self.data_type = dict(
            mock='mock', 
            single='pc' if (use_camera if use_camera is not None else config.use_camera) else 'obj_pc', 
            # seperate='',  NotImplemented
            sparseconv='voxel', 
        )[config.actor.vision_backbone_type]
    
    def reset(self, obs_dict, indices=None):
        if indices is None:
            indices = torch.arange(len(obs_dict['dof_pos']), device=obs_dict['dof_pos'].device)
        # self.last_qpos[indices] = obs_dict['dof_pos'][indices]
        # self.last_qpos_action[indices] = obs_dict['dof_pos'][indices]
        # self.last_vel[indices] = 0
    
    def query(self, obs_dict):
        robot_state = torch.cat([obs_dict['dof_pos']], dim=-1)
        if True:#self.data_type == 'obj_pc':
            robot_state = torch.cat([robot_state, obs_dict['obj_trans']], dim=-1)
            if self.config.actor.with_rot:
                robot_state = torch.cat([robot_state, obs_dict['rel_goal_rot'].reshape(*robot_state.shape[:-1], 9)], dim=-1)
            visual_observation = obs_dict['obj_pc']
        elif self.data_type == 'pc':
            visual_observation = obs_dict['pc']
        elif self.data_type == 'voxel':
            coordinates_list = []
            features_list = []
            for i in range(len(obs_dict['pc'])):
                coordinates, features = ME.utils.sparse_quantize(
                    coordinates=obs_dict['pc'][i, :, :3],
                    features=obs_dict['pc'][i],
                    quantization_size=self.config.quantization_size, 
                )
                coordinates_list.append(coordinates)
                features_list.append(features)
            coordinates, features = ME.utils.sparse_collate(
                coordinates_list, 
                features_list, 
            )
            visual_observation = ME.TensorField(
                coordinates=coordinates, 
                features=features, 
                device=robot_state.device, 
            )
        else:
            visual_observation = torch.zeros(len(robot_state), 0, dtype=torch.float, device=robot_state.device)
        result = dict(robot_state_stacked=robot_state[:, None], visual_observation=visual_observation)
        result = torch.cat([robot_state, visual_observation.reshape(len(robot_state), -1)], dim=-1)
        return result

    def update(self, obs_dict, action):
        return
        # self.last_qpos[:] = obs_dict['dof_pos']
        # self.last_vel[:] = obs_dict['dof_vel']
        # self.last_qpos_action[:] = action