import os
import sys
import multiprocessing

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.realpath('.'))

from tqdm import tqdm
import numpy as np
import torch
import ikpy.chain
# from torchprimitivesdf.sdf import box_sdf
from utils.robot_model import RobotModel

def box_sdf(points, extents):
    q = points.abs() - extents
    return (torch.maximum(q, torch.zeros_like(q)).square().sum(dim=-1) + 1e-9).sqrt() + torch.minimum(q.max(dim=-1).values, torch.zeros_like(q.max(dim=-1).values))

def init():
    global arm
    arm = ikpy.chain.Chain.from_urdf_file('data/ur_description/urdf/ur5_simplified.urdf', active_links_mask=[False, True, True, True, True, True, True, False])

def direct(bias, old_arm_pose):
    global arm
    return _direct(bias, old_arm_pose, arm)

def _direct(bias, old_arm_pose, arm):
    if bias == 0:
        return old_arm_pose
    arm_qpos = np.zeros(8)
    arm_qpos[1:7] = old_arm_pose
    fks = arm.forward_kinematics(arm_qpos)
    fks[2,3] += bias
    ik = arm.inverse_kinematics(target_position=fks[:3, 3], 
                                             target_orientation=fks[:3, :3], 
                                             orientation_mode='all',
                                             initial_position=arm_qpos,
                                             )
    return ik[1:7]
    
class SafetyWrapper():
    def __init__(self, p1, p2, safety_dilation, thres_safety, max_iter, step_size, silent=True, robot_pc_points=0):
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
        self.robot_model = RobotModel(
            urdf_path='data/ur_description/urdf/ur5_leap_simplified.urdf', 
            n_surface_points=robot_pc_points, 
            device=p1.device, 
        )
        self.device = p1.device
        self.arm = ikpy.chain.Chain.from_urdf_file('data/ur_description/urdf/ur5_simplified.urdf', active_links_mask=[False, True, True, True, True, True, True, False])
        self.pool = multiprocessing.Pool(10, initializer=init)
        
    def direct(self, actions):
        # return actions
        origin_pose = torch.tensor([[0,0,0,1,0,0,0,1,0]], dtype=actions.dtype, device=actions.device).repeat(len(actions), 1)
        robot_pose = torch.cat([origin_pose, actions], dim=-1)
        self.robot_model.set_parameters(robot_pose)
        collision_vertices = self.robot_model.get_collision_vertices(self.link_names+["upper_arm_link"])
        torch_bias = torch.clamp(0.002+self.box_origin[-1]-self.box_extents[-1]-collision_vertices[..., 2].min(dim=1).values, min=0)
        bias = torch_bias.cpu().numpy()
        old_arm_pose = actions[:, :6].cpu().numpy()
        # new_arm_pose = []
        # for i in range(len(actions)):
        #     new_arm_pose.append(_direct(bias[i], old_arm_pose[i], self.arm))
        new_arm_pose = self.pool.starmap(direct, [(bias[i], old_arm_pose[i]) for i in range(len(actions))])
        new_arm_pose = torch.tensor(new_arm_pose, dtype=actions.dtype, device=actions.device)
        return torch.cat([new_arm_pose, actions[:, 6:]], dim=-1), torch_bias
    
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
