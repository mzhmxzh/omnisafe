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
import gc

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

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
        self.wrapper = SafetyWrapper(config, torch.tensor([-1,-1,ROBOT_BASE_HEIGHT], device=self.device), torch.tensor([1,1,1], device=self.device), 0, 0, 0, 0, robot_pc_points=config.robot_pc_points)
        self.is_corrector = self.config.get("is_corrector", 0)
        # create_isaac_visualizer(port=6000, host="localhost", keep_default_viewer=True, max_env=4)
        
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
        self.robot_asset = robot_asset
        self.robot_rb_count = self.gym.get_asset_rigid_body_count(robot_asset) + 1
        self.hand_idx = self.gym.find_asset_rigid_body_index(robot_asset, 'hand_base_link')
        self.fingertip_names = ['thumb_fingertip', 'fingertip', 'fingertip_2', 'fingertip_3']
        self.fingertip_idx = [self.gym.find_asset_rigid_body_index(robot_asset, name) for name in self.fingertip_names]
        self.cforces_idx = []
        self.fforces_idx = []
        self.global_forces_idx = []

        # table_idx = self.gym.find_asset_rigid_body_index(table_asset, "box")
        # sensor_pose = gymapi.Transform(gymapi.Vec3(0.0,0.0,0.0))
        # sensor_props = gymapi.ForceSensorProperties()
        # sensor_props.enable_constraint_solver_forces = True
        # sensor_props.enable_forward_dynamics_forces = False
        # sensor_props.use_world_frame = True
        # sensor_idx2 = self.gym.create_asset_force_sensor(table_asset, table_idx, sensor_pose, sensor_props)
        # self.table_force_idx.append(sensor_idx2)
        for name in ['hand_base_link', 'mcp_joint', 'pip', 'dip', 'fingertip', 'mcp_joint_2', 'pip_2', 'dip_2', 'fingertip_2', 'mcp_joint_3', 'pip_3', 'dip_3', 'fingertip_3', 'pip_4', 'thumb_pip', 'thumb_dip', 'thumb_fingertip']:
            body_idx = self.gym.find_asset_rigid_body_index(robot_asset, name)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0,0.0,0.0))
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_constraint_solver_forces = False
            sensor_props.enable_forward_dynamics_forces = True
            sensor_props.use_world_frame = False
            sensor_idx1 = self.gym.create_asset_force_sensor(robot_asset, body_idx, sensor_pose, sensor_props)
            self.cforces_idx.append(sensor_idx1)

            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_constraint_solver_forces = True
            sensor_props.enable_forward_dynamics_forces = False
            sensor_props.use_world_frame = True
            sensor_idx1 = self.gym.create_asset_force_sensor(robot_asset, body_idx, sensor_pose, sensor_props)
            self.global_forces_idx.append(sensor_idx1)

            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_constraint_solver_forces = True
            sensor_props.enable_forward_dynamics_forces = False
            sensor_props.use_world_frame = False
            sensor_idx2 = self.gym.create_asset_force_sensor(robot_asset, body_idx, sensor_pose, sensor_props)
            self.fforces_idx.append(sensor_idx2)


        
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
            if self.config.use_camera:
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
                
                assert len(traj) > 0, object_code
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
        self.obj_original_masses = torch.tensor(self.obj_original_masses, device=self.device, dtype=torch.float)

        self.gym.prepare_sim(self.sim)
        self.robot_state_tensor_initial = torch.stack([HOME_POSE, HOME_POSE*0], dim=1).to(self.device)
        self.robot_state_tensor_initial = self.robot_state_tensor_initial[None, :, :].repeat(config.num_envs, 1, 1)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_state_tensor_initial))
        self.robot_state_tensor_initial_save = self.robot_state_tensor_initial.clone()
        self.target = self.robot_state_tensor_initial[:,:,0].clone()
        self.unsafe = torch.zeros((self.num_envs,), device=self.device)
        self.rollback_buf = torch.zeros((self.num_envs,), device=self.device).bool()
        self.unsafe_table = torch.zeros((self.num_envs,), device=self.device)
        self.unsafe_object = torch.zeros((self.num_envs,), device=self.device)
        self.unsafe_fingers = torch.zeros((self.num_envs,), device=self.device)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rigid_body_tensor = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.rb_forces = torch.zeros((self.num_envs, rigid_body_tensor.shape[1], 3), dtype=torch.float, device=self.device).squeeze()

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
        net_contact_force_tensor_raw = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_tensor_raw = self.gym.acquire_dof_force_tensor(self.sim)
        force_sensor_raw = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.net_contact_force_tensor = gymtorch.wrap_tensor(net_contact_force_tensor_raw).reshape(self.num_envs,-1,3)
        self.force_tensor = gymtorch.wrap_tensor(force_tensor_raw)
        self.force_sensor = gymtorch.wrap_tensor(force_sensor_raw)
        self.global_z_sensor_read = torch.zeros_like(self.force_sensor).reshape(self.num_envs,-1,6).float()
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
        # self.palm_bias = torch.tensor(PALM_BIAS, device=self.device, dtype=torch.float)
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
        self.thr_z = self.config.get("thr_z", 10.0)
        self.thr_x = self.config.get("thr_x", 10.0)
        self.thr_obj_force = self.config.get("thr_obj_force", 10.0)

        self.cost = torch.zeros((self.num_envs),device=self.device)
        
        # self.gym = bind_visualizer_to_gym(self.gym, self.sim)
        # set_gpu_pipeline(True)
    
    def _reset(self, obj_trans=None, obj_rot=None, init_qpos=None, indices=None):
        self.rb_forces[indices, :, :] = 0.0
        self.record_success[indices] = self.check_success()[indices]
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
        bias[:, :2] += torch.rand_like(bias[:, :2]) * self.config.rand_radius
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
        for index in indices:
            self.robot_state_tensor_initial[index.item(),:,1] = 0
        self.robot_indices_flat = self.robot_indices[indices].to(torch.int32).reshape(-1)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.robot_state_tensor_initial),
                                              gymtorch.unwrap_tensor(self.robot_indices_flat), len(self.robot_indices_flat))
        self.target[indices] = self.robot_state_tensor_initial[indices,:,0].clone()
        self.progress_buf[indices] = -self.config.init_timesteps
        for index in indices:
            self.goal[index.item()] = 0
        if self.config.actor.with_rot:
            self.goal_rot[indices] = pttf.random_rotations(len(indices), device=self.device)
    
    def reset(self, obj_trans=None, obj_rot=None, init_qpos=None, indices=None):
        self._reset(obj_trans=obj_trans, obj_rot=obj_rot, init_qpos=init_qpos, indices=indices)
        for i in range(self.config.init_timesteps):
            self.step(coordinator_prob=0)
        return self.get_state(coordinator_prob=0)

    def safety_wrapper(self, actions, execute=True):
        return self.wrapper.direct(actions, execute)
    
    def step(self, actions=None, goal=None, step_cnt=-1, with_delay=False, mask_RL=None, coordinator_prob=None, non_RL_idx=None):
        if step_cnt > 0:
            self.curr_iter = step_cnt
            if self.config.physics_randomization:
                if self.curr_iter % self.config.randomize_adr_every == 0:
                    self.need_rerandomize = torch.ones_like(self.need_rerandomize, device=self.device).bool()

        self.refresh()

        # apply random force
        if self.config.force_scale > 0.0:
            self.rb_forces *= (0.99 ** (0.1 / 0.08))
            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.config.random_force_prob).nonzero().reshape(-1)
            #force_indices = torch.logical_and(force_indices, self.progress_buf >= 0)
            rb_force_shape = self.rb_forces[force_indices, self.robot_rb_count:].shape
            rb_force_scale = torch.randn((rb_force_shape[0],rb_force_shape[1]), device=self.device)
            rb_force_dir = torch.randn(rb_force_shape, device=self.device)
            rb_force_dir = rb_force_dir / rb_force_dir.norm(dim=-1, keepdim=True)
            # proportional to mass
            rb_force_scale = rb_force_scale.unsqueeze(-1)
            self.rb_forces[force_indices, self.robot_rb_count:, :] = rb_force_dir * self.obj_original_masses[force_indices].reshape(rb_force_shape[0],-1,1) * (self.config.force_scale)
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None,
                                                    gymapi.LOCAL_SPACE)

        # self.clone_backup_states()

        # pre-pseudo wrapper: check table penetration
        # if actions is not None:
        #     lift_mask = (self.progress_buf >= 0)
        #     lift_idx = torch.arange(0, self.num_envs, device=self.device, dtype=torch.long)[lift_mask]
        #     self.target[lift_idx], self.unsafe[lift_idx], _ = self.safety_wrapper(actions[lift_idx].clone(),lift_idx,pseudo=True)

        #     # if self.config.physics_randomization and with_delay:
        #     #     self.adr_last_action_queue, self.target = update_delay_queue(self.adr_last_action_queue, self.target, self.adr_cfg)

        # pseudo-execution
        self.prev_dof_pos = self.dof_state_tensor[:, :, 0].clone()
        self.global_z_sensor_read = self.force_sensor.reshape(self.num_envs, -1, 6)[:, self.global_forces_idx, 2].sum(dim=1).clone()
        assert self.global_z_sensor_read.dim() == 1, self.global_z_sensor_read.shape
        z_sensor_min = self.force_sensor.reshape(self.num_envs, -1, 6)[:, self.global_forces_idx, 2].sum(dim=1).min(dim=0)[0].clone() + 10000
        z_sensor_max = self.force_sensor.reshape(self.num_envs, -1, 6)[:, self.global_forces_idx, 2].sum(dim=1).max(dim=0)[0].clone() - 10000
        for step in range(self.substeps):
            # linear interpolation. Subaction represneted as target pos
            subactions = self.prev_dof_pos + (self.target - self.prev_dof_pos) * (step+1) / self.substeps
            self.substep(subactions)
            sub_global_z_sensor_read = self.force_sensor.reshape(self.num_envs, -1, 6)[:, self.global_forces_idx, 2].sum(dim=1).clone()
            self.global_z_sensor_read = torch.where(sub_global_z_sensor_read > self.global_z_sensor_read, sub_global_z_sensor_read, self.global_z_sensor_read)
            sub_z_sensor_min = self.force_sensor.reshape(self.num_envs, -1, 6)[:, self.global_forces_idx, 2].sum(dim=1).min(dim=0)[0].clone()
            sub_z_sensor_max = self.force_sensor.reshape(self.num_envs, -1, 6)[:, self.global_forces_idx, 2].sum(dim=1).max(dim=0)[0].clone()
            z_sensor_min = torch.where(sub_z_sensor_min < z_sensor_min, sub_z_sensor_min, z_sensor_min)
            z_sensor_max = torch.where(sub_z_sensor_max > z_sensor_max, sub_z_sensor_max, z_sensor_max)
        # print(self.global_z_sensor_read.mean())
        # print("force sensor on hand")
        # print(z_sensor_min)
        # print(z_sensor_max)
            
        # post-pseudo wrapper: check obj/finger penetration
        # if actions is not None:
        #     lift_mask = (self.progress_buf >= 0)
        #     lift_idx = torch.arange(0, self.num_envs, device=self.device, dtype=torch.long)[lift_mask]
        #     self.target[lift_idx], self.rollback_buf[lift_idx], _ = self.safety_wrapper(actions[lift_idx].clone(),lift_idx,pseudo=False)

            # if self.config.physics_randomization and with_delay:
            #     self.adr_last_action_queue, self.target = update_delay_queue(self.adr_last_action_queue, self.target, self.adr_cfg)

        # rollback envs with unsafe contact, essentially skip one step of execution.
        # Thus Dagger does not see unsafe actions.
        # print(self.rollback_buf.sum())
        # if self.rollback_buf.sum() > 0:
        #     self.rollback(self.rollback_buf)

        # # if ad-hoc lifting is needed for unsafe envs, implement here
        # self.prev_dof_pos = self.dof_state_tensor[:, :, 0].clone()
        # for step in range(self.substeps):
        #     # linear interpolation. Subaction represneted as target pos
        #     subactions = self.prev_dof_pos + (self.target - self.prev_dof_pos) * (step+1) / self.substeps
        #     self.substep(subactions)
        
        state = self.get_state(coordinator_prob=coordinator_prob, non_RL_idx=non_RL_idx)

        self.obj_init_trans = torch.where(self.progress_buf[:, None] == -1, state['obj_trans'], self.obj_init_trans)
        self.obj_init_rot = torch.where(self.progress_buf[:, None, None] == -1, state['obj_rot'], self.obj_init_rot)
        if goal is not None and goal.shape[0] > 0:
            self.goal[self.progress_buf == -1] = goal[self.progress_buf == -1]

        self.progress_buf += 1

        # re-randomize physics
        if self.config.physics_randomization:
            rerandomize_physics_gravity(self, step_cnt)
        self.refresh()
        self.cost = self.cal_safety_cost(actions) if actions is not None else torch.zeros((self.num_envs),device=self.device)
        return state
    
    def get_state(self,coordinator_prob=None, non_RL_idx=None):
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
        result_dict['obj_pc'] = torch.einsum('nab,nkb->nka', obj_rot_mat, self.obj_full_pcs[self.env_ids.long(), self.current_obj_idx]) + obj_trans[:, None]
        result_dict['reward'], reward_detail_dict = self.compute_reward(result_dict,coordinator_prob, non_RL_idx)
        result_dict['reward_detail_names'] = list(reward_detail_dict.keys())
        result_dict.update(reward_detail_dict)
        result_dict['success'] = self.check_success()
        result_dict['goal'] = self.goal.clone() if self.config.actor.with_goal else None
        if self.config.use_camera:
            #t = time.time()
            result_dict['pc'] = collect_pointclouds(self.gym, self.sim, [fv[self.current_obj_idx[i]] for i, fv in enumerate(self.face_verts)], obj_trans, obj_rot_mat, result_dict['obj_pc'].clone(), self.robot_model, result_dict['dof_pos'], self.rigid_body_state_tensor[:, self.hand_idx, :3], self.config.num_envs, self.progress_buf[0].item(), self.camera_props, self.camera_u2, self.camera_v2, self.env_origin, self.camera_depth_tensor_list, self.camera_rgb_tensor_list, self.camera_seg_tensor_list, self.camera_vinv_tensor_list, self.camera_proj_tensor_list, self.device)
            #print(f"camera time:{time.time() - t}")
        if self.use_adr:
            result_dict['available'] = torch.logical_and(result_dict['available'], (self.worker_types == RolloutWorkerModes.ADR_ROLLOUT))
            result_dict['npd'] = self.npd 
            result_dict['boundary_sr'] = self.boundary_sr
            result_dict['adr_ranges'] = self.adr_ranges

        # TODO: implement cost
        result_dict['cost'] = self.cost
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
        # step the physics
        self.gym.simulate(self.sim)
        self.refresh()
    
    def compute_reward(self, result_dict, coordinator_prob=None, non_RL_idx=None):
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
        # obj_unmoved = (-5 * (obj_trans - obj_init_trans).norm(dim=-1) - 0.5 * pttf.so3_relative_angle(obj_rot_mat, self.obj_init_rot, eps=1e-3)).exp()
        mp_goal_reward = -(self.goal - result_dict['dof_pos']).abs().mean(dim=-1)
        pen_safety = self.unsafe

        # safety metrics
        unsafe_all = self.unsafe
        unsafe_table = self.unsafe_table
        unsafe_object = self.unsafe_object
        unsafe_fingers = self.unsafe_fingers
        
        # multi-contact & palm contact
        # right_hand_palm_pos = torch.stack([get_point_on_link(self.rigid_body_state_tensor[:, self.hand_idx], self.palm_bias[i]) for i in range(len(self.palm_bias))], dim=-2)
        # right_hand_palm_dists = (right_hand_palm_pos - obj_trans[:, None]).norm(dim=-1)
        # contact_palm = [points_in_contact(obj_pc, right_hand_palm_pos[:,i], torch.einsum('nab,bc->nac', pttf.quaternion_to_matrix(self.rigid_body_state_tensor[:, self.hand_idx, [6, 3, 4 ,5]]), self.rigid_body_rot['hand_base_link']), self.rigid_body_size['palm'], 0.005) for i in range(len(self.palm_bias))]
        # contact_palm_reward = torch.stack(contact_palm).float().mean(dim=0).tanh()
        # contact_rich = contact['thumb_fingertip'] * (contact['fingertip'] + contact['fingertip_2'] + contact['fingertip_3']) > 2
        # TODO: calculate orientation
        
        # TODO: define reward
        if self.config.staged_reward or self.config.reward_type == "staged":
            raise NotImplementedError('is_close_enough not implemented now because there are no target to use')
            reward = (reach_reward * 0.5 * (1-0.95*is_contact) + contact_reward * 0.5 * is_close_enough + (lift_reward * 5 + obj_dis_reward * 20) * is_contact - action_pen * w_action - w_tpen * tpen - pen_safety * 20).clamp(min=-1)/5
        elif self.config.reward_type == "staged2":
            # reward = (reach_reward * 0.5 * (1-0.95*is_contact) + contact_reward * 0.5 + (lift_reward * 5 + obj_dis_reward * 20)/5 * is_contact + (contact_rich + contact_palm_reward) * is_contact - action_pen * w_action - w_tpen * tpen - pen_safety * 20).clamp(min=-1)/5
            reward = (reach_reward * 0.5 * (1-0.95*is_contact) + contact_reward * 0.5 + (lift_reward * 5 + obj_dis_reward * 20)/5 * is_contact - action_pen * w_action - w_tpen * tpen - pen_safety * 20).clamp(min=-1)/5
        elif self.config.reward_type == "not_staged":
            reward = (reach_reward * 0.5 + contact_reward * 0.5 + lift_reward * 5 + obj_dis_reward * 20 - action_pen * w_action - w_tpen * tpen - pen_safety * 20).clamp(min=-1)/5
        elif self.config.reward_type == "reorientation":
            raise NotImplementedError
        else:
            raise NotImplementedError

        reward_detail_dict.update(dict(tpen=tpen, real_obj_height=real_obj_height,obj_dis_reward=obj_dis_reward,reach_reward=reach_reward, contact_reward=contact_reward, lift_reward=lift_reward, action_pen=action_pen,
                                    unsafe_all=unsafe_all,unsafe_table=unsafe_table,unsafe_object=unsafe_object,unsafe_fingers=unsafe_fingers))

        return reward, reward_detail_dict

    def refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
    
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

    def clone_backup_states(self):
        self.backup_root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).clone()
        self.backup_dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).clone()
        # self.backup_dof_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)).clone()
        self.backup_rigid_body_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).clone()
        # self.backup_env_rb_states = [self.gym.get_env_rigid_body_states(self.envs[i],gymapi.STATE_ALL) for i in range(self.num_envs)]
        # self.backup_net_contact_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).clone()
        # self.backup_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)).clone()
        # self.backup_jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobine_tensor(self.sim)).clone()

    def rollback(self,env_mask):
        indices = torch.arange(self.num_envs, device=self.device)[env_mask]
        # print("c1")
        indices_flat = self.obj_indices[indices].to(torch.int32).reshape(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.backup_root_state_tensor),
                                            gymtorch.unwrap_tensor(indices_flat), len(indices_flat))
        # print("c2")
        indices_flat = self.robot_indices[indices].to(torch.int32).reshape(-1)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.backup_dof_state_tensor),
                                            gymtorch.unwrap_tensor(indices_flat), len(indices_flat))
        # print("c3")
        # for i in indices:
        #     self.gym.set_env_rigid_body_states(self.envs[i], self.backup_env_rb_states[i], gymapi.STATE_ALL)
        # print("c4")
        self.rollback_buf = (self.rollback_buf > 1)
        # print("c5")

    def cal_safety_cost(self, actions):
        # case1: table_penetration
        cost_tpen = torch.zeros((self.num_envs), device=self.device).bool()
        cost_tpen[self.progress_buf >= 0] = self.safety_wrapper(actions[self.progress_buf >= 0].clone(),execute=False)

        # case2: downward contact force
        cost_object = torch.zeros((self.num_envs), device=self.device).bool()
        cost_tpen[self.progress_buf >= 0] = (self.global_z_sensor_read > self.thr_obj_force)[self.progress_buf >= 0]

        # case3: finger contact
        cost_finger = torch.zeros((self.num_envs), device=self.device).bool()
        fingertip_indices = [4,8,12,16]
        fforce_fingertip = self.force_sensor.reshape(self.num_envs, -1, 6)[:, self.fforces_idx, :3][:,fingertip_indices,:]
        fforce_max_env = fforce_fingertip.max(dim=1)[0]
        unsafe_mask = torch.logical_or(fforce_max_env[:,0] > self.thr_x, fforce_max_env[:,2] > self.thr_z)
        cost_finger[self.progress_buf >= 0] = unsafe_mask[self.progress_buf >= 0]

        cost = torch.logical_and(cost_tpen, cost_object)
        cost = torch.logical_and(cost, cost_finger)
        return cost.float()