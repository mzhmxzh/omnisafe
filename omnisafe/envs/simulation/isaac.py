"""
Description: Wrapper for Isaac Gym
"""

import os
import sys
from scipy.stats import loguniform

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.realpath('.'))

import argparse
import numpy as np
import transforms3d
from isaacgym import gymapi, gymtorch

import torch
import random
import warnings
from pytorch3d import transforms as pttf
# from torchsdf.sdf import index_vertices_by_faces
from utils.robot_model import RobotModel
from utils.isaac_utils import joint_names_isaac, joint_names, get_sim_params, get_plane_params, get_table_asset_options, get_robot_asset_options, get_object_asset_options, create_table_actor, create_robot_actor, init_waiting_pose, get_camera_params, load_cameras, collect_pointclouds, get_point_on_link, points_in_contact, update_sim_params
from utils.robot_info import ROBOT_BASE_HEIGHT, TABLE_HEIGHT, HOME_POSE, RIGID_BODY_BIAS, RIGID_BODY_SIZE, RIGID_BODY_RPY, OBJ_INIT_CENTER, OLD_HOME_POSE
from utils.control import PIDController
from utils.config import load_config
from utils.adr_utils import get_property_setter_map, get_property_getter_map, get_default_setter_args, generate_random_samples, apply_random_samples, update_delay_queue, init_adr_params, update_adr_param_and_rerandomize, update_manualdr_param_and_rerandomize, rerandomize_physics_gravity
from utils.safety_wrapper import SafetyWrapper
import trimesh as tm
from tqdm import trange
from collections import deque
from copy import deepcopy
import time
import ikpy.chain

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'j1', 'j0', 'j2', 'j3', 
    'j12', 'j13', 'j14', 'j15', 
    'j5', 'j4', 'j6', 'j7', 
    'j9', 'j8', 'j10', 'j11', 
]

def eps_dis(dis, eps=0):
    return (dis-eps).clip(min=0)

class RolloutWorkerModes:
    ADR_ROLLOUT = 0 # rollout with current ADR params
    ADR_BOUNDARY = 1 # rollout with params on boundaries of ADR, used to decide whether to expand ranges
    TEST_ENV = 2 # rollout wit default DR params, used to measure overall success rate. (currently unused)


class Env():
    def __init__(self, config):
        self.config = config
        self.adr_cfg = self.config.get("adr", {})
        self.use_adr = self.adr_cfg.get("use_adr", False)
        self.from_start = config.get('from_start', 0)
        self.curr_iter = 0
        self.robot_model = RobotModel(
            urdf_path='data/ur_description/urdf/ur5_leap_simplified.urdf', 
            n_surface_points=config.robot_pc_points, 
            device=config.gpu, 
        )
        self.arm = ikpy.chain.Chain.from_urdf_file('data/ur_description/urdf/ur5_simplified.urdf', active_links_mask=[False, True, True, True, True, True, True, False])
        self.root_dir = 'data'
        self.device = torch.device('cpu') if config.gpu=='cpu' else torch.device(f'cuda:{config.gpu}')
        self.wrapper = SafetyWrapper(torch.tensor([-1,-1,ROBOT_BASE_HEIGHT], device=self.device), torch.tensor([1,1,1], device=self.device), 0, 0, 0, 0, robot_pc_points=config.robot_pc_points)
        
        assert config.env_hz % config.control_hz == 0
        self.substeps = config.env_hz // config.control_hz

        self.gym = gymapi.acquire_gym()
        sim_params = get_sim_params(config)
        self.sim = self.gym.create_sim(config.gpu, config.gpu, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
        
        plane_params = get_plane_params()
        self.gym.add_ground(self.sim, plane_params)

        table_asset_options = get_table_asset_options()
        table_dims = gymapi.Vec3(2, 2, TABLE_HEIGHT)
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)

        robot_asset_options = get_robot_asset_options()
        robot_asset = self.gym.load_urdf(self.sim, self.root_dir, 'ur_description/urdf/ur5_leap_isaac.urdf', robot_asset_options)
        self.hand_idx = self.gym.find_asset_rigid_body_index(robot_asset, 'hand_base_link')
        self.fingertip_names = ['thumb_fingertip', 'fingertip', 'fingertip_2', 'fingertip_3']
        self.fingertip_idx = [self.gym.find_asset_rigid_body_index(robot_asset, name) for name in self.fingertip_names]
        
        if self.use_adr:
            init_adr_params(self, config, self.adr_cfg)

        object_asset_options = get_object_asset_options()
        self.num_envs = config.num_envs
        self.num_obj_per_env = config.num_obj_per_env
        if len(config.object_code) != self.num_obj_per_env * self.num_envs:
            warnings.warn(f'Number of objects in split ({len(config.object_code)}) does not match number of environments ({config.num_envs}) times number of objects per environment ({config.num_obj_per_env}).', UserWarning)

        self.env_objects = [[] for _ in range(self.num_envs)]
        obj_raw_pcs = [[] for _ in range(self.num_envs)]
        self.face_verts = [[] for _ in range(self.num_envs)]
        if config.load_traj:
            self.loaded_traj = [[] for _ in range(self.num_envs)]
            self.current_traj = [None for _ in range(self.num_envs)]
            self.grasp_trans = torch.zeros((self.num_envs, 3), device=self.device)
            self.grasp_rot = torch.zeros((self.num_envs, 3, 3), device=self.device)
            self.grasp_qpos = torch.zeros((self.num_envs, 16), device=self.device)

        scale2float_dict = {
            '006': 0.06,
            '008': 0.08,
            '010': 0.10,
            '012': 0.12,
            '015': 0.15,
            '100': 1.00,
        }

        random.shuffle(config.object_code)
        loaded_assets = dict()
        self.obj_names = [[] for _ in range(self.num_envs)]
        for i in trange(self.num_obj_per_env * self.num_envs):
            object_code = config.object_code[i%len(config.object_code)]
            if object_code in loaded_assets.keys():
                this_asset = loaded_assets[object_code]
            else:
                this_asset = self.gym.load_asset(self.sim, os.path.join(config.asset_path, 'meshdata', object_code[:object_code.rfind('_')], 'coacd'), f'coacd_{object_code[object_code.rfind("_")+1:]}.urdf', object_asset_options)
                # this_asset = self.gym.load_asset(self.sim, os.path.join(config.asset_path, 'meshdata', object_code[:object_code.rfind('_')], 'coacd'), f'coacd_{object_code[object_code.rfind("_")+1:]}_sdf.urdf', object_asset_options)
                loaded_assets[object_code] = this_asset
            self.env_objects[i // self.num_obj_per_env].append(this_asset)
            raw_pc = np.load(os.path.join(config.asset_path, 'meshdata', object_code[:object_code.rfind('_')], 'coacd', 'fps.npy')).astype(np.float32)
            obj_raw_pcs[i // self.num_obj_per_env].append(raw_pc*scale2float_dict[object_code.split('_')[-1]])
            object_mesh = tm.load(os.path.join(self.root_dir, 'meshdata', object_code[:object_code.rfind('_')], 'coacd', 'decomposed.obj'), process=False).apply_scale(scale2float_dict[object_code.split('_')[-1]])
            v = torch.tensor(object_mesh.vertices, dtype=torch.float, device=self.device)
            f = torch.tensor(object_mesh.faces, dtype=torch.long, device=self.device)
            self.face_verts[i // self.num_obj_per_env].append(v[f])
            self.obj_names[i // self.num_obj_per_env].append(object_code)
            if config.load_traj:
                traj_path = os.path.join(self.root_dir, 'results_filtered-v15', object_code + '.npy')
                if os.path.exists(traj_path):
                    traj = np.load(traj_path, allow_pickle=True)
                else:
                    raise NotImplementedError
                    traj = []
                self.loaded_traj[i // self.num_obj_per_env].append(traj)
        self.obj_full_pcs = torch.from_numpy(np.stack([np.stack(raw_pcs) for raw_pcs in obj_raw_pcs])).to(self.device)
        obj_waiting_pose = init_waiting_pose(config.num_obj_per_env, table_dims, config.env_spacing, config.object_rise)

        if config.use_camera:
            self.camera_depth_tensor_list = []
            self.camera_rgb_tensor_list = []
            self.camera_seg_tensor_list = []
            self.camera_vinv_tensor_list = []
            self.camera_proj_tensor_list = []
            self.env_origin = torch.zeros((config.num_envs, 3), device=self.device)
            self.camera_props = get_camera_params()
            camera_eye = torch.tensor(config.camera.xyz, device=self.device)
            camera_lookat = torch.tensor(config.camera.eye, device=self.device)
            camera_eye[2] += ROBOT_BASE_HEIGHT + TABLE_HEIGHT
            camera_lookat[2] += ROBOT_BASE_HEIGHT + TABLE_HEIGHT
            camera_u = torch.arange(0, self.camera_props.width)
            camera_v = torch.arange(0, self.camera_props.height)
            camera_v2, camera_u2 = torch.meshgrid(camera_v, camera_u)
            self.camera_v2, self.camera_u2 = camera_v2.to(self.device), camera_u2.to(self.device)

        robot_indices = []
        obj_indices = []
        all_indices = []
        self.envs = []
        self.obj_handles = []
        self.obj_original_masses = []
        self.robot_handles = []

        for i in trange(config.num_envs):
            max_agg_bodies = self.gym.get_asset_rigid_body_count(robot_asset) * 1 + 2 * sum([self.gym.get_asset_rigid_body_count(object_asset) for object_asset in self.env_objects[i]]) + 1
            max_agg_shapes = self.gym.get_asset_rigid_shape_count(robot_asset) * 1 + 2 * sum([self.gym.get_asset_rigid_shape_count(object_asset) for object_asset in self.env_objects[i]]) + 1
            env = self.gym.create_env(self.sim, gymapi.Vec3(-config.env_spacing, -config.env_spacing, 0), gymapi.Vec3(config.env_spacing, config.env_spacing, config.env_spacing), 6)
            self.gym.begin_aggregate(env, max_agg_bodies, max_agg_shapes, True)

            lower_limits, upper_limits, robot_actor_handle, robot_props = create_robot_actor(self.gym, env, robot_asset, table_dims, robot_indices, all_indices, config.friction)
            self.original_dof_prop = robot_props
            self.robot_handles.append(robot_actor_handle)

            create_table_actor(self.gym, env, self.sim, table_asset, table_dims, all_indices)

            obj_indices.append([])
            self.obj_handles.append([])
            self.obj_original_masses.append([])
            for j in range(config.num_obj_per_env):
                pose = obj_waiting_pose[j]
                object_actor_handle = self.gym.create_actor(env, self.env_objects[i][j], pose, "object", 0, 1, 2)
                obj_indices[-1].append(self.gym.get_actor_index(env, object_actor_handle, gymapi.DOMAIN_SIM))
                all_indices.append(self.gym.get_actor_index(env, object_actor_handle, gymapi.DOMAIN_SIM))
                object_shape_props = self.gym.get_actor_rigid_shape_properties(env, object_actor_handle)
                object_shape_props[0].friction = config.friction
                assert object_shape_props[0].restitution == 0.0, object_shape_props[0].restitution
                self.gym.set_actor_rigid_shape_properties(env, object_actor_handle, object_shape_props)
                # for randomization use
                self.obj_handles[-1].append(object_actor_handle)
                object_body_props = self.gym.get_actor_rigid_body_properties(env, object_actor_handle)
                object_shape_props = self.gym.get_actor_rigid_shape_properties(env, object_actor_handle)
                self.obj_original_masses[-1].append(object_body_props[0].mass)
                self.obj_original_restitution = object_shape_props[0].restitution
                assert self.obj_original_restitution == 0.0, self.obj_original_restitution
            
            if config.use_camera:
                load_cameras(self.gym, self.sim, env, self.device, i, self.camera_props, camera_eye, camera_lookat, self.camera_depth_tensor_list, self.camera_rgb_tensor_list, self.camera_seg_tensor_list, self.camera_vinv_tensor_list, self.camera_proj_tensor_list, self.env_origin)

            self.gym.end_aggregate(env)
            self.envs.append(env)

        self.env_ids = torch.arange(0, config.num_envs, device=self.device, dtype=torch.int32)
        self.robot_indices = torch.tensor(robot_indices, device=self.device, dtype=torch.int32)
        self.obj_indices = torch.tensor(obj_indices, device=self.device, dtype=torch.long)
        self.all_indices = torch.tensor(all_indices, device=self.device, dtype=torch.int32)
        self.lower_limits = torch.tensor(lower_limits, device=self.device, dtype=torch.float)
        self.upper_limits = torch.tensor(upper_limits, device=self.device, dtype=torch.float)

        self.gym.prepare_sim(self.sim)
        self.robot_state_tensor_initial = torch.stack([HOME_POSE, HOME_POSE*0], dim=1).to(self.device)
        self.robot_state_tensor_initial = self.robot_state_tensor_initial[None, :, :].repeat(config.num_envs, 1, 1)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_state_tensor_initial))
        self.robot_state_tensor_initial_save = self.robot_state_tensor_initial.clone()
        self.target = self.robot_state_tensor_initial[:,:,0].clone()

        if config.use_viewer:
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 75
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
            cam_pos = gymapi.Vec3(0.7, 1.0, 0.35921312 + table_dims.z)
            cam_target = gymapi.Vec3(0.7, 0, 0.29278688 + table_dims.z)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            print('created viewer')

        root_state_tensor_raw = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor_raw = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state_tensor_raw = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(root_state_tensor_raw)
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.dof_state_tensor = gymtorch.wrap_tensor(dof_state_tensor_raw).reshape(config.num_envs, -1, 2)
        self.rigid_body_state_tensor = gymtorch.wrap_tensor(rigid_body_state_tensor_raw).reshape(config.num_envs, -1, 13)
        # object falls from a height
        self.obj_init_state = torch.tensor([OBJ_INIT_CENTER[0].item(), OBJ_INIT_CENTER[1].item(), config.object_rise+table_dims.z,0,0,0,1,0,0,0,0,0,0], device=self.device)
        self.current_obj_idx = torch.full((config.num_envs,), -1, dtype=torch.long, device=self.device)
        self.progress_buf = torch.full_like(self.current_obj_idx, -config.init_timesteps)
        self.record_success = torch.zeros_like(self.current_obj_idx, dtype=torch.bool)
        self.record_angle = torch.full_like(self.current_obj_idx, np.pi, dtype=torch.float32)
        self.rigid_body_bias = {k: torch.tensor(v, device=self.device, dtype=torch.float) for k, v in RIGID_BODY_BIAS.items()}
        self.rigid_body_rot = {k: torch.tensor(transforms3d.euler.euler2mat(*v), device=self.device, dtype=torch.float) for k, v in RIGID_BODY_RPY.items()}
        self.rigid_body_size = {k: torch.tensor(v, device=self.device, dtype=torch.float) for k, v in RIGID_BODY_SIZE.items()}
        self.obj_init_trans = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_init_rot = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.obj_init_rot += torch.eye(3, device=self.device)
        self.rel_goal_rot = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.rel_goal_rot += torch.tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.goal_rot += torch.eye(3, device=self.device)
        self.goal = torch.zeros((self.num_envs, 22), device=self.device)
        self.tpen = torch.zeros((self.num_envs,), device=self.device)

        # simuate grasp
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)

        self.controller = PIDController(0)
        
        # self.gym = bind_visualizer_to_gym(self.gym, self.sim)
        # set_gpu_pipeline(True)
    
    def _reset(self, obj_trans=None, obj_rot=None, init_qpos=None, indices=None):
        if indices is None:
            indices = torch.arange(0, self.num_envs, device=self.device, dtype=torch.long)
        self.record_success[indices] = self.check_success()[indices]
        self.record_angle[indices] = pttf.so3_rotation_angle(self.rel_goal_rot[indices], eps=1e-3)
        arange = torch.arange(0, len(indices), device=self.device, dtype=torch.long)
        if len(indices) == 0:
            return

        # ADR enqueue and randomization
        if self.use_adr:
            update_adr_param_and_rerandomize(self, indices, RolloutWorkerModes)
        elif self.config.physics_randomization:
            update_manualdr_param_and_rerandomize(self, indices)
        
        self.current_obj_idx[indices] = (self.current_obj_idx[indices] + 1) % self.num_obj_per_env
        if self.config.load_traj and not self.from_start:
            for i in indices:
                self.current_traj[i] = random.choice(self.loaded_traj[i][self.current_obj_idx[i]]) 

        self.obj_indices_flat = self.obj_indices[indices].to(torch.int32).reshape(-1)
        self.root_state_tensor[self.obj_indices_flat.long()] = self.saved_root_tensor[self.obj_indices_flat.long()]

        bias = torch.zeros((len(indices), 3), device=self.device)
        bias[:, :2] = OBJ_INIT_CENTER.to(self.device)
        bias[:, :2] += (torch.rand_like(bias[:, :2]) * 2 - 1) * self.config.rand_radius
        bias[:, 2] -= ROBOT_BASE_HEIGHT
        rot_angle = (torch.rand((len(indices), ), device=self.device) - 0.5) * np.pi # rotate at most 90 degrees
        rot_matrix = torch.eye(3, device=self.device).repeat(len(indices), 1, 1)
        rot_matrix[:, 0, 0] = torch.cos(rot_angle)
        rot_matrix[:, 0, 1] = -torch.sin(rot_angle)
        rot_matrix[:, 1, 0] = torch.sin(rot_angle)
        rot_matrix[:, 1, 1] = torch.cos(rot_angle)
        
        if obj_trans is not None:
            obj_trans[:, 2] += TABLE_HEIGHT + ROBOT_BASE_HEIGHT
        elif self.config.load_traj and not self.from_start:
            obj_trans = torch.from_numpy(np.array([self.current_traj[i]['object_pose'][:3, 3] for i in indices])).to(self.device).float()
            obj_trans += bias
            obj_trans[:, 2] += TABLE_HEIGHT + ROBOT_BASE_HEIGHT
        else:
            obj_trans = torch.zeros((len(indices), 3), device=self.device)
            obj_trans[:, :2] = OBJ_INIT_CENTER.to(self.device)
            rand_angle = torch.rand((len(indices), ), device=self.device) * 2 * np.pi
            rand_length = torch.sqrt(torch.rand((len(indices),), device=self.device)) * self.config.rand_radius
            obj_trans[:, 0] += rand_length * torch.cos(rand_angle)
            obj_trans[:, 1] += rand_length * torch.sin(rand_angle)
            # object falls from a height
            obj_trans[:, 2] += TABLE_HEIGHT + self.config.object_rise
        self.root_state_tensor[self.obj_indices[indices][arange, self.current_obj_idx[indices]], :3] = obj_trans

        if obj_rot is not None:
            obj_rot = pttf.matrix_to_quaternion(obj_rot.to(self.device))[:, [1, 2, 3, 0]]
        elif self.config.load_traj and not self.from_start:
            obj_rot = torch.from_numpy(np.stack([self.current_traj[i]['object_pose'][:3, :3] for i in indices], axis=0)).float().to(self.device)
            obj_rot = torch.einsum('nab,nbc->nac', rot_matrix, obj_rot)
            obj_rot = pttf.matrix_to_quaternion(obj_rot)[:, [1, 2, 3, 0]]
        else:
            obj_rot = pttf.random_quaternions(len(indices), device=self.device)
        self.root_state_tensor[self.obj_indices[indices][arange, self.current_obj_idx[indices]], 3:7] = obj_rot

        if self.config.load_traj:
            pass
            # self.grasp_trans[indices] = torch.tensor([[self.current_traj[i]['hand']['final_grasp'][name] for name in translation_names] for i in indices]).to(self.device).float()
            # self.grasp_rot[indices] = torch.tensor([transforms3d.euler.euler2mat(*[self.current_traj[i]['hand']['final_grasp'][name] for name in rot_names]) for i in indices]).to(self.device).float()
            # self.grasp_qpos[indices] = torch.tensor([[self.current_traj[i]['hand']['final_grasp'][name] for name in joint_names] for i in indices]).to(self.device).float()

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                            gymtorch.unwrap_tensor(self.obj_indices_flat), len(self.obj_indices_flat))

        if init_qpos is not None:
            self.robot_state_tensor_initial[indices,:,0] = init_qpos
        elif self.config.load_traj and not self.from_start:
            cpu_bias = bias.cpu().numpy()
            cpu_rot = rot_matrix.cpu().numpy()
            for num, i in enumerate(indices):
                hand_obj_trans = np.array([self.current_traj[i]['qpos'][name] for name in translation_names])
                hand_obj_rot = np.array(transforms3d.euler.euler2mat(*[self.current_traj[i]['qpos'][name] for name in rot_names]))
                hand_rot = np.einsum('ab,bc,cd->ad', cpu_rot[num], self.current_traj[i]['object_pose'][:3, :3], hand_obj_rot)
                hand_trans = self.current_traj[i]['object_pose'][:3, 3] + np.einsum('ab,bc,c->a', cpu_rot[num], self.current_traj[i]['object_pose'][:3, :3], hand_obj_trans)
                hand_trans += cpu_bias[num]
                arm_qpos = self.arm.inverse_kinematics(
                    target_position=hand_trans, 
                    target_orientation=hand_rot, 
                    orientation_mode='all', 
                    initial_position=[0]+OLD_HOME_POSE[:6].tolist()+[0], 
                )[1:7]
                arm_qpos = np.array(arm_qpos)
                hand_qpos = np.array([self.current_traj[i]['qpos'][name] for name in joint_names])
                qpos = np.concatenate((arm_qpos, hand_qpos))
                self.robot_state_tensor_initial[i, :, 0] = torch.from_numpy(qpos).float().to(self.device)
        else:
            self.robot_state_tensor_initial[indices] = self.robot_state_tensor_initial_save[indices]
        self.robot_state_tensor_initial[indices,:,1] = 0
        self.robot_indices_flat = self.robot_indices[indices].to(torch.int32).reshape(-1)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.robot_state_tensor_initial),
                                              gymtorch.unwrap_tensor(self.robot_indices_flat), len(self.robot_indices_flat))
        self.target[indices] = self.robot_state_tensor_initial[indices,:,0].clone()
        self.progress_buf[indices] = -self.config.init_timesteps
        self.goal[indices] = 0
        if self.config.actor.with_rot:
            self.goal_rot[indices] = pttf.random_rotations(len(indices), device=self.device)
    
    def reset(self, obj_trans=None, obj_rot=None, init_qpos=None, indices=None):
        self._reset(obj_trans=obj_trans, obj_rot=obj_rot, init_qpos=init_qpos, indices=indices)
        for i in range(self.config.init_timesteps):
            self.step()
        return self.get_state()

    def safety_wrapper(self, actions):
        return self.wrapper.direct(actions)
    
    def step(self, actions=None, goal=None, step_cnt=-1, with_delay=False):
        if step_cnt > 0:
            self.curr_iter = step_cnt
            if self.config.physics_randomization:
                if self.curr_iter % self.config.randomize_adr_every == 0:
                    self.need_rerandomize = torch.ones_like(self.need_rerandomize, device=self.device).bool()

        self.refresh()

        if actions is not None:
            self.target[self.progress_buf >= 0], self.tpen[self.progress_buf >= 0] = self.safety_wrapper(actions[self.progress_buf >= 0].clone())
            if self.config.physics_randomization and with_delay:
                self.adr_last_action_queue, self.target = update_delay_queue(self.adr_last_action_queue, self.target, self.adr_cfg)
        self.prev_dof_pos = self.dof_state_tensor[:, :, 0].clone()

        for step in range(self.substeps):
            # linear interpolation. Subaction represneted as target pos
            subactions = self.prev_dof_pos + (self.target - self.prev_dof_pos) * (step+1) / self.substeps
            self.substep(subactions)
        state = self.get_state()

        self.obj_init_trans = torch.where(self.progress_buf[:, None] == -1, state['obj_trans'], self.obj_init_trans)
        self.obj_init_rot = torch.where(self.progress_buf[:, None, None] == -1, state['obj_rot'], self.obj_init_rot)
        if goal is not None and goal.shape[0] > 0:
            self.goal[self.progress_buf == -1] = goal[self.progress_buf == -1]
            # self.robot_model.set_parameters_simple(self.goal[self.progress_buf == -1])
            # self.goal_ft[self.progress_buf == -1] = self.robot_model.get_fingertips()
        #     need_goal = (self.progress_buf == -1).nonzero().reshape(-1)
        #     self.goal[need_goal] = goal
        #     arm_qpos = np.zeros((len(need_goal), 8))
        #     arm_qpos[:, 1:7] = goal[:, :6].cpu().numpy()
        #     fks = np.zeros((len(need_goal), 4, 4))
        #     for i, q in enumerate(arm_qpos):
        #         fks[i] = self.arm.forward_kinematics(q)
        #     fks = torch.from_numpy(fks).to(torch.float).to(goal.device)
        #     self.grasp_trans[need_goal] = torch.einsum('nba,nb->na', state['obj_rot'][need_goal], fks[:, :3, 3] - state['obj_trans'][need_goal])
        #     self.grasp_rot[need_goal] = torch.einsum('nba,nbc->nac', state['obj_rot'][need_goal], fks[:, :3, :3])
        #     self.grasp_qpos[need_goal] = goal[:, 6:]
        self.progress_buf += 1

        # re-randomize physics
        if self.config.physics_randomization:
            rerandomize_physics_gravity(self, step_cnt)
        self.refresh()
        return state
    
    def get_state(self):
        self.refresh()
        result_dict = {}
        result_dict['done'] = (self.progress_buf == self.config.act_timesteps-2)
        result_dict['available'] = torch.logical_and((self.progress_buf >= 0), (self.progress_buf < self.config.act_timesteps-1-8))
        if self.config.actor.get('joint', dict()).get('type', "") == 'goal':
            result_dict['available'] = torch.logical_and(result_dict['available'], (self.choice == 1))
        result_dict['progress_buf'] = self.progress_buf.float() / 50
        result_dict['dof_pos'] = self.dof_state_tensor[:, :, 0].clone()
        result_dict['dof_vel'] = self.dof_state_tensor[:, :, 1].clone()
        obj_trans = self.root_state_tensor[self.obj_indices[self.env_ids.long(), self.current_obj_idx], :3].clone()
        obj_trans[:, 2] -= TABLE_HEIGHT + ROBOT_BASE_HEIGHT
        obj_rot_quat = self.root_state_tensor[self.obj_indices[self.env_ids.long(), self.current_obj_idx], 3:7].clone()
        obj_rot_mat = pttf.quaternion_to_matrix(obj_rot_quat[:, [3, 0, 1, 2]])
        result_dict['obj_trans'] = obj_trans
        result_dict['obj_rot'] = obj_rot_mat
        result_dict['goal_rot'] = self.goal_rot.clone()
        hand_rot_quat = self.rigid_body_state_tensor[:, self.hand_idx, 3:7].clone()
        hand_rot_mat = pttf.quaternion_to_matrix(hand_rot_quat[:, [3, 0, 1, 2]])
        obj_rel_hand = torch.einsum('nab,ncb->nac', obj_rot_mat, hand_rot_mat)
        self.rel_goal_rot = torch.einsum('nab,ncb->nac', self.goal_rot, obj_rel_hand)
        result_dict['rel_goal_rot'] = self.rel_goal_rot.clone()
        # print('time: ', self.progress_buf[0])
        # print('obj')
        # print(obj_rot_mat[0])
        # print('hand')
        # print(hand_rot_mat[0])
        # print('obj_rel_hand')
        # print(obj_rel_hand[0])
        # print('goal')
        # print(self.goal_rot[0])
        # print('rel_goal')
        # print(self.rel_goal_rot[0])
        # print(pttf.so3_rotation_angle(self.rel_goal_rot[[0]])[0]*180/np.pi)
        # time.sleep(0.2)
        result_dict['obj_pc'] = torch.einsum('nab,nkb->nka', obj_rot_mat, self.obj_full_pcs[self.env_ids.long(), self.current_obj_idx]) + obj_trans[:, None]
        result_dict['reward'], reward_detail_dict = self.compute_reward(result_dict)
        result_dict['reward_detail_names'] = list(reward_detail_dict.keys())
        result_dict.update(reward_detail_dict)
        result_dict['success'] = self.check_success()
        result_dict['goal'] = self.goal.clone() if self.config.actor.with_goal else None
        if self.config.use_camera:
            #t = time.time()
            result_dict['pc'] = collect_pointclouds(self.gym, self.sim, [fv[self.current_obj_idx[i]] for i, fv in enumerate(self.face_verts)], obj_trans, obj_rot_mat, result_dict['obj_pc'].clone(), self.robot_model, result_dict['dof_pos'], self.rigid_body_state_tensor[:, self.hand_idx, :3], self.config.num_envs, self.progress_buf[0].item(), self.camera_props, self.camera_u2, self.camera_v2, self.env_origin, self.camera_depth_tensor_list, self.camera_rgb_tensor_list, self.camera_seg_tensor_list, self.camera_vinv_tensor_list, self.camera_proj_tensor_list, self.device)
            #print(f"camera time:{time.time() - t}")
        if self.use_adr:
            if self.curr_iter < 20000:
                result_dict['available'] = torch.logical_and(result_dict['available'], (self.worker_types == RolloutWorkerModes.ADR_ROLLOUT))
            result_dict['npd'] = self.npd 
            result_dict['boundary_sr'] = self.boundary_sr
        # TODO: calculate cost
        return result_dict

    def substep(self, subactions):
        self.refresh()
        dof_pos = self.dof_state_tensor[:, :, 0].clone()
        # calculate the (compensated) traget position for control
        self.temp_target = (self.controller.step(dof_pos, subactions)).to(torch.float)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.temp_target))
        # update the viewer
        if self.config.use_viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                self.close()
                exit(0)
        self.gym.fetch_results(self.sim, True)
        if self.config.use_viewer or self.config.use_camera:
            self.gym.step_graphics(self.sim)
            if self.config.use_viewer:
                self.gym.draw_viewer(self.viewer, self.sim, False)
        # gym.sync_frame_time(sim)
        # step the physics
        self.gym.simulate(self.sim)
    
    def compute_reward(self, result_dict):
        obj_pc = result_dict['obj_pc'].clone()
        real_obj_height = obj_pc[..., 2].min(dim=-1).values + ROBOT_BASE_HEIGHT
        obj_pc[..., 2] += ROBOT_BASE_HEIGHT + TABLE_HEIGHT
        obj_trans = self.root_state_tensor[self.obj_indices[self.env_ids.long(), self.current_obj_idx], :3].clone()
        # obj_rot = self.root_state_tensor[self.obj_indices[self.env_ids.long(), self.current_obj_idx], 3:7].clone()
        # obj_rot_mat = pttf.quaternion_to_matrix(obj_rot[:, [3, 0, 1, 2]])
        obj_height = (obj_trans[:, 2] - TABLE_HEIGHT - ROBOT_BASE_HEIGHT - self.obj_init_trans[:, 2]).clamp(min=-0.1)
        obj_goal = self.obj_init_trans.clone()
        obj_goal[:, 2] += 0.15 + TABLE_HEIGHT + ROBOT_BASE_HEIGHT
        obj_dis = (obj_trans - obj_goal).norm(dim=-1)
        reward_detail_dict = dict(obj_height=obj_height, obj_dis=obj_dis)

        # right_hand_pos = get_point_on_link(self.rigid_body_state_tensor[:, self.hand_idx], self.rigid_body_bias['hand_base_link'])
        right_hand_finger_pos = torch.stack([get_point_on_link(self.rigid_body_state_tensor[:, self.fingertip_idx[i]], self.rigid_body_bias[name]) for i, name in enumerate(self.fingertip_names)], dim=-2)
        # distance from palm root to object center
        # right_hand_dist = (right_hand_pos - obj_trans).norm(dim=-1).clamp(max=0.5)
        right_hand_finger_dists = (right_hand_finger_pos - obj_trans[:, None]).norm(dim=-1)
        # distance from fingertip to object center
        # right_hand_finger_dist = right_hand_finger_dists.sum(dim=-1).clamp(max=1.6)

        finger_weight = torch.full((4,), 0.01, device=self.device)
        finger_weight[self.fingertip_names.index('thumb_fingertip')] = 0.04
        reach_reward = (finger_weight / (0.06 + right_hand_finger_dists.clamp(min=0.03,max=0.8))).sum(dim=-1)
        contact = {name: points_in_contact(obj_pc, right_hand_finger_pos[:, i], torch.einsum('nab,bc->nac', pttf.quaternion_to_matrix(self.rigid_body_state_tensor[:, self.fingertip_idx[i], [6, 3, 4 ,5]]), self.rigid_body_rot[name]), self.rigid_body_size[name], 0.005) for i, name in enumerate(self.fingertip_names)}
        contact_reward = contact['thumb_fingertip'] * (contact['fingertip'] + contact['fingertip_2'] + contact['fingertip_3']).tanh()
        obj_init_trans = self.obj_init_trans.clone()
        obj_init_trans[:, 2] += TABLE_HEIGHT + ROBOT_BASE_HEIGHT
        # lift_reward = contact_reward * (0.6 - obj_dis).clamp(min=5e-2)
        lift_reward = contact_reward * obj_height.clamp(min=0, max=0.2).sqrt()
        # init_dis = (obj_init_trans - obj_goal).norm(dim=-1)
        # lift_reward = contact_reward * (init_dis - obj_dis).clamp(min=1e-6).sqrt()
        # lift_reward = contact_reward * ((0.2-(obj_height-0.2).abs()).clamp(min=0)+1e-6).sqrt()
        bonus = (contact_reward != 0).float() * (real_obj_height > 0.01).float()
        obj_dis_reward = bonus * (-(obj_dis*10).square()).exp()

        # palm_state = self.rigid_body_state_tensor[:, self.hand_idx]
        # palm_trans = palm_state[:, :3]
        # palm_rot = pttf.quaternion_to_matrix(palm_state[:, [6, 3, 4, 5]])
        # hand_qpos = self.dof_state_tensor[:, 6:, 0]
        
        # palm_state = self.rigid_body_state_tensor[:, self.hand_idx]
        # palm_trans = palm_state[:, :3]
        # palm_rot = pttf.quaternion_to_matrix(palm_state[:, [6, 3, 4, 5]])
        # hand_qpos = self.dof_state_tensor[:, 6:, 0]

        # robot_state = result_dict['dof_pos']
        # robot_translation = torch.zeros([len(robot_state), 3], dtype=torch.float, device=robot_state.device)
        # robot_rotation = torch.eye(3, dtype=torch.float, device=robot_state.device).expand(len(robot_state), 3, 3)
        # robot_pose = torch.cat([
        #     robot_translation,
        #     robot_rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        #     robot_state[:, :22], 
        # ], dim=-1)
        # self.robot_model.set_parameters(robot_pose)
        # robot_pc_fps = self.robot_model.get_surface_points(nobase=True)
        # min_height = robot_pc_fps[..., 2].min(dim=-1).values + ROBOT_BASE_HEIGHT
        tpen = (self.tpen).square().clamp(max=20)

        # palm_trans_target = torch.einsum('nab,nb->na', obj_rot_mat, self.grasp_trans) + obj_trans
        # palm_rot_target = torch.einsum('nab,nbc->nac', obj_rot_mat, self.grasp_rot)

        # check stage of RL rollout
        # is_close_enough = (torch.norm(palm_trans-palm_trans_target, dim=-1) < 0.2)
        #is_close_enough = (torch.norm(palm_trans-self.obj_init_trans, dim=-1) < 0.2)
        is_contact = (contact_reward>0)
        # is_lift = (real_obj_height > 0.01)

        # palm rotation and translation
        # trans_dis =  (palm_trans-palm_trans_target).square().sum(dim=-1)
        # rot_dis = pttf.so3_relative_angle(palm_rot, palm_rot_target, eps=1e-3).square()
        # qpos_dis = (hand_qpos - self.grasp_qpos).square().mean(dim=-1)
        # goal_reward = -(trans_dis * 2.5 +
        #                 rot_dis * 0.03 + 
        #                 qpos_dis * 0.5)
        
        # penalize velocity
        action_pen = result_dict['dof_vel'].clamp(min=-1,max=1).square().mean(dim=-1)
        w_action = 10 if self.config.strict_constraint else self.config.get("w_action", 0.5)
        w_tpen = 5 if self.config.strict_constraint else 1

        rel_angle = pttf.so3_rotation_angle(result_dict['rel_goal_rot'], eps=1e-3)
        
        if self.config.staged_reward or self.config.reward_type == "staged":
            raise NotImplementedError('is_close_enough not implemented now because there are no target to use')
            reward = (reach_reward * 0.5 * (1-0.95*is_contact) + contact_reward * 0.5 * is_close_enough + (lift_reward * 5 + obj_dis_reward * 20) * is_contact - action_pen * w_action - w_tpen * tpen).clamp(min=-1)/5
        elif self.config.reward_type == "staged2":
            reward = (reach_reward * 0.5 * (1-0.95*is_contact) + contact_reward * 0.5 + (lift_reward * 5 + obj_dis_reward * 20) * is_contact - action_pen * w_action - w_tpen * tpen).clamp(min=-1)/5
        elif self.config.reward_type == "staged2_rot":
            reward = (reach_reward * 0.2 + contact_reward * (0.5 + (np.pi - rel_angle)) - action_pen * w_action - w_tpen * tpen)/5
        elif self.config.reward_type == "not_staged":
            reward = (reach_reward * 0.5 + contact_reward * 0.5 + lift_reward * 5 + obj_dis_reward * 20 - action_pen * w_action - w_tpen * tpen).clamp(min=-1)/5
        else:
            raise NotImplementedError

        reward_detail_dict.update(dict(record_angle=self.record_angle*180/np.pi, rel_angle=rel_angle*180/np.pi, tpen=tpen, real_obj_height=real_obj_height,obj_dis_reward=obj_dis_reward,reach_reward=reach_reward, contact_reward=contact_reward, lift_reward=lift_reward, action_pen=action_pen))

        return reward, reward_detail_dict

    def refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
    
    def randomize_start(self):
        self.progress_buf = torch.randint(-self.config.init_timesteps, self.config.act_timesteps, (self.num_envs,), device=self.device)

    def check_success(self):
        obj_trans = self.root_state_tensor[self.obj_indices[self.env_ids.long(), self.current_obj_idx], :3].clone()
        obj_lift_height = (obj_trans[:, 2] - TABLE_HEIGHT - ROBOT_BASE_HEIGHT - self.obj_init_trans[:, 2]).clamp(min=-0.1)
        success = obj_lift_height > 0.1
        if self.config.actor.with_rot:
            rel_angle = pttf.so3_rotation_angle(self.rel_goal_rot, eps=1e-3)
            success = (rel_angle * 180 / np.pi < 15)
        return success

    def close(self):
        if self.config.use_viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        
    def modify_adr_param(self, param, direction, adr_param_dict, param_limit=None):
        op = adr_param_dict["delta_style"]
        delta = adr_param_dict["delta"]
        
        if direction == 'up':

            if op == "additive":
                new_val = param + delta
            elif op == "scaling":
                assert delta > 1.0, "Must have delta>1 for multiplicative ADR update."
                new_val = param * delta
            else:
                raise NotImplementedError

            assert param_limit is not None, adr_param_dict
            if param_limit is not None:
                new_val = min(new_val, param_limit)
            
            changed = abs(new_val - param) > 1e-9
            
            return new_val, changed
        
        elif direction == 'down':

            if op == "additive":
                new_val = param - delta
            elif op == "scaling":
                assert delta > 1.0, "Must have delta>1 for multiplicative ADR update."
                new_val = param / delta
            else:
                raise NotImplementedError

            assert param_limit is not None, adr_param_dict
            if param_limit is not None:
                new_val = max(new_val, param_limit)
            
            changed = abs(new_val - param) > 1e-9
            
            return new_val, changed
        else:
            raise NotImplementedError
    
    def apply_randomization(self, ind, adr_params, boundary=False):
        # sample and set all parameters that are randomized
        env = self.envs[ind]
        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in adr_params.items():
            # actor = ['robot','object']
            # actor = dict of specific params to randomize
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                # prop_name = 'friction', 'mass', etc
                # prop_attrs = dict of attribute and value pairs
                if prop_name == 'scale':
                    # TODO: when randomizing scale, need to scale the point cloud of objects accordingly.
                    attr_randomization_params = prop_attrs
                    sample = generate_random_samples(attr_randomization_params, 1,
                                                        self.curr_iter, None)
                    og_scale = 1.0
                    if attr_randomization_params['delta_style'] == 'scaling':
                        new_scale = og_scale * sample
                    elif attr_randomization_params['delta_style'] == 'additive':
                        new_scale = og_scale + sample
                    if not self.gym.set_actor_scale(env, handle, new_scale): # TODO: the original scale is not 1 now
                        raise ValueError(f"set scale failed: actor={actor}, actor_properties={actor_properties}")
                    continue
                # if list it is likely to be 
                #  - rigid_body_properties
                #  - rigid_shape_properties
                prop = param_getters_map[prop_name](env, handle)
                if isinstance(prop, list):
                    if self.adr_first_randomize:
                        self.original_props[prop_name] = [
                            {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                    for attr, attr_randomization_params in prop_attrs.items():
                        # if boundary and (not (attr_randomization_params['range_sampling'][1] == attr_randomization_params['range_sampling'][0])):
                        #     continue
                        for body_idx, (p, og_p) in enumerate(zip(prop, self.original_props[prop_name])):
                            curr_prop = p 
                            apply_random_samples(
                                curr_prop, og_p, attr, attr_randomization_params,
                                self.curr_iter, None,)
                # if it is not a list, it is likely an array 
                # which means it is for dof_properties
                else:
                    if self.adr_first_randomize:
                        self.original_props[prop_name] = deepcopy(prop)
                    for attr, attr_randomization_params in prop_attrs.items():
                        # if boundary and (not (attr_randomization_params['range_sampling'][1] == attr_randomization_params['range_sampling'][0])):
                        #     continue
                        apply_random_samples(
                                prop, self.original_props[prop_name], attr,
                                attr_randomization_params, self.curr_iter, None)
                setter = param_setters_map[prop_name]
                default_args = param_setter_defaults_map[prop_name]
                setter(env, handle, prop)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--exp_config', type=str, default='config/isaac_lds_new.yaml')
    parser.add_argument('--object_code', type=list)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--use_viewer', type=int)
    args = parser.parse_args()

    args = load_config(args.exp_config, args)

    env = Env(args)
    env.reset()
    while True:
        env.step()
