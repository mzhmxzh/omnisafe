import os
import sys
import multiprocessing

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.realpath('.'))

from tqdm import tqdm
import numpy as np
import torch
import ikpy.chain
from collections import OrderedDict
from isaacgym import gymapi, gymtorch
from pytorch3d import transforms as pttf
# from torchprimitivesdf.sdf import box_sdf
# from pykin.kinematics.transform import Transform
# from pykin.robots.single_arm import SingleArm
# from pykin.utils import plot_utils as p_utils
from utils.robot_model import RobotModel
# from utils.robot_info import OLD_HOME_POSE

# init_thetas = OLD_HOME_POSE[:6].cpu().numpy().tolist()

def box_sdf(points, extents):
    q = points.abs() - extents
    return (torch.maximum(q, torch.zeros_like(q)).square().sum(dim=-1) + 1e-9).sqrt() + torch.minimum(q.max(dim=-1).values, torch.zeros_like(q.max(dim=-1).values))

def init():
    global arm
    arm = ikpy.chain.Chain.from_urdf_file('data/ur_description/urdf/ur5_simplified.urdf', active_links_mask=[False, True, True, True, True, True, True, False])
    # arm = SingleArm("urdf/ur5/ur5_simplified.urdf", Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
    # arm.setup_link_name("base_link", "hand_base_link")

def direct(bias, old_arm_pose):
    global arm
    return _direct(bias, old_arm_pose, arm)

def _direct(bias, old_arm_pose, arm):
    if bias == 0:
        return (old_arm_pose, np.array([0]))
    arm_qpos = np.zeros(8)
    arm_qpos[1:7] = old_arm_pose
    fks = arm.forward_kinematics(arm_qpos)
    fks[2,3] += bias
    ik = arm.inverse_kinematics(target_position=fks[:3, 3], 
                                             target_orientation=fks[:3, :3], 
                                             orientation_mode='all',
                                             initial_position=arm_qpos,
                                             )
    return (ik[1:7], np.array([1]))
    
class SafetyWrapper():
    def __init__(self, config, p1, p2, safety_dilation, thres_safety, max_iter, step_size, silent=True, robot_pc_points=0):
        self.config = config
        self.box_origin = (p2 + p1) / 2
        self.box_extents = (p2 - p1) / 2
        self.safety_dilation = safety_dilation
        self.thres_safety = thres_safety
        self.max_iter = max_iter
        self.step_size = step_size
        self.silent = silent
        self.link_names = [
            'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 
            'hand_base_link', 
            'mcp_joint', 'pip', 'dip', 'fingertip', 
            'mcp_joint_2', 'pip_2', 'dip_2', 'fingertip_2', 
            'mcp_joint_3', 'pip_3', 'dip_3', 'fingertip_3', 
            'pip_4', 'thumb_pip', 'thumb_dip', 'thumb_fingertip', 
        ]
        self.selected_asset_linkname = OrderedDict([
            ('dip','j2'),('fingertip','j3'),('dip_2','j6'),('fingertip_2','j7'),
            ('dip_3','j10'),('fingertip_3','j11'),('thumb_dip','j14'),('thumb_fingertip','j15')
        ])

        self.robot_model = RobotModel(
            urdf_path='data/ur_description/urdf/ur5_leap_simplified.urdf', 
            n_surface_points=robot_pc_points, 
            device=p1.device, 
        )
        self.joint_order = self.robot_model.joint_order.copy()
        self.device = p1.device
        self.arm = ikpy.chain.Chain.from_urdf_file('data/ur_description/urdf/ur5_simplified.urdf', active_links_mask=[False, True, True, True, True, True, True, False])
        self.pool = multiprocessing.Pool(10, initializer=init)
        self.gravity_dir = torch.tensor([[0.],[0.],[-1.]],device=self.device, dtype=torch.float)
        self.obj_bias = 0.02
        self.rigid_body_dict ={'base_link': 0, 'dip': 10, 'dip_2': 18, 'dip_3': 22, 'fingertip': 11, 
        'fingertip_2': 19, 'fingertip_3': 23, 'forearm_link': 3, 'hand_base_link': 7, 'mcp_joint': 8, 
        'mcp_joint_2': 16, 'mcp_joint_3': 20, 'pip': 9, 'pip_2': 17, 'pip_3': 21, 'pip_4': 12, 
        'shoulder_link': 1, 'thumb_dip': 14, 'thumb_fingertip': 15, 'thumb_pip': 13, 'upper_arm_link': 2, 
        'wrist_1_link': 4, 'wrist_2_link': 5, 'wrist_3_link': 6}
        self.fingertip_rigid_bodies = [11,19,23,15]
        
    def direct(self, actions):
        # return actions
        origin_pose = torch.tensor([[0,0,0,1,0,0,0,1,0]], dtype=actions.dtype, device=actions.device).repeat(len(actions), 1)
        robot_pose = torch.cat([origin_pose, actions], dim=-1)
        self.robot_model.set_parameters(robot_pose)
        collision_vertices = self.robot_model.get_collision_vertices(self.link_names+["upper_arm_link"])
        torch_bias = torch.clamp(0.002+self.box_origin[-1]-self.box_extents[-1]-collision_vertices[..., 2].min(dim=1).values, min=0)
        bias = torch_bias.cpu().numpy()
        old_arm_pose = actions[:, :6].cpu().numpy()

        direct_return = self.pool.starmap(direct, [(bias[i], old_arm_pose[i]) for i in range(len(actions))])
        new_arm_pose = [r[0] for r in direct_return]
        unsafe = [r[1] for r in direct_return]
        new_arm_pose = torch.tensor(new_arm_pose, dtype=actions.dtype, device=actions.device)
        unsafe = torch.tensor(unsafe, dtype=actions.dtype, device=actions.device).squeeze()
        return torch.cat([new_arm_pose, actions[:, 6:]], dim=-1), unsafe, torch_bias

    def direct_object(self, isaac, actions, lift_idx):
        assert len(actions) == len(lift_idx), len(actions)
        # read world-frame force sensor values
        # min_sensor_read = torch.minimum(min_sensor_read, self.force_sensor.reshape(self.num_envs, -1, 6)[:, self.fforces_idx, :3].sum(dim=1).min(dim=0)[0])
        # max_sensor_read = torch.maximum(max_sensor_read, self.force_sensor.reshape(self.num_envs, -1, 6)[:, self.fforces_idx, :3].sum(dim=1).max(dim=0)[0])
        # print("force sensor on hand")
        # print(min_sensor_read)
        # print(max_sensor_read)


        # unsafe = total force z value > noise_thr = 5N
        unsafe_mask = (isaac.global_z_sensor_read > isaac.config.force_z_noise)[lift_idx]
        assert unsafe_mask.dim() == 1, unsafe_mask.shape
        unsafe_idx = torch.arange(len(actions),device=actions.device)[unsafe_mask]

        # lift hand with direct()
        # old_arm_pose = isaac.dof_state_tensor[:,:6, 0].clone().cpu().numpy()
        old_arm_pose = actions[:, :6].cpu().numpy()
        new_arm_pose = torch.tensor(old_arm_pose, dtype=actions.dtype, device=actions.device)
        if len(unsafe_idx) > 0:
            direct_return = self.pool.starmap(direct, [(self.obj_bias, old_arm_pose[i]) for i in unsafe_idx])
            ik_return_pose = torch.tensor([r[0] for r in direct_return], dtype=actions.dtype, device=actions.device)
            new_arm_pose[unsafe_idx] = ik_return_pose
        return torch.cat([new_arm_pose, actions[:, 6:]], dim=-1), unsafe_mask

    def direct_fingers(self, isaac, actions, lift_idx, current_pos=None, thr_x=10.0, thr_z=10.0):
        assert len(actions) == len(lift_idx), len(actions)

        fingertip_indices = [4,8,12,16]
        fforce_fingertip = isaac.force_sensor.reshape(isaac.num_envs, -1, 6)[:, isaac.fforces_idx, :3][:,fingertip_indices,:]
        fforce_max_env = fforce_fingertip.max(dim=1)[0]
        unsafe_mask = torch.logical_or(fforce_max_env[:,0].abs() > thr_x, fforce_max_env[:,2].abs() > thr_z)[lift_idx]

        unsafe_idx = torch.arange(len(actions),device=actions.device)[unsafe_mask]
        if len(unsafe_idx) == 0:
            return actions, unsafe_mask

        # reset hand contact force
        if current_pos is None:
            current_pos = isaac.backup_dof_state_tensor[unsafe_idx,:,0].clone() #set target_pos = current_pos
        actions[unsafe_idx,...] = current_pos
        return actions, unsafe_mask
    
    def cal_E_safety(self, action):
        robot_pose = torch.cat([
            -self.box_origin.expand(len(action), 3), 
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device=self.device).expand(len(action), 6), 
            action, 
        ], dim=1)
        self.robot_model.set_parameters(robot_pose)
        collision_vertices = self.robot_model.get_collision_vertices(self.link_names)
        # dis_local, dis_signs, _ = box_sdf(collision_vertices.reshape(-1, 3), self.box_extents - self.safety_dilation)
        # dis_local = (dis_local + 1e-8).sqrt()
        # dis_local = torch.where(dis_signs, dis_local, -dis_local)
        # dis_local = dis_local.reshape(collision_vertices.shape[0], collision_vertices.shape[1])
        dis_local = box_sdf(collision_vertices.reshape(-1, 3), self.box_extents - self.safety_dilation)
        dis_local = dis_local.reshape(collision_vertices.shape[0], collision_vertices.shape[1])
        dis_local[dis_local < 0] = 0
        E_safety = dis_local.sum(dim=-1)
        return E_safety
    
    def __call__(self, action):
        if len(action.shape) == 1:
            single = True
            action = action.unsqueeze(0)
        else:
            single = False
        action = action.detach().requires_grad_()
        E_safety = self.cal_E_safety(action)
        if E_safety.max() < self.thres_safety:
            return action.detach()
        for step in range(1, self.max_iter) if self.silent else tqdm(range(1, self.max_iter + 1), desc='safety', total=self.max_iter):
            E_safety.sum().backward()
            with torch.no_grad():
                action[:, :6] = action[:, :6] - action.grad[:, :6] * self.step_size
            action.grad.data.zero_()
            E_safety = self.cal_E_safety(action)
            if E_safety.max() < self.thres_safety:
                break
        assert E_safety.max() < self.thres_safety
        if single:
            action = action[0]
        return action.detach()
