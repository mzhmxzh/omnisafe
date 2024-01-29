import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.realpath('.'))

import json
from collections import OrderedDict
import numpy as np
import transforms3d
import trimesh as tm
import torch
import yaml

from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.robot_info import FINGERTIP_BIAS, FINGERTIP_NORMAL
from urdf_parser_py.urdf import Robot, Box, Mesh
import plotly.graph_objects as go
import pytorch3d.structures
import pytorch3d.ops
try:
    from torchprimitivesdf import box_sdf, transform_points_inverse, fixed_transform_points_inverse
except:
    print('torchprimitivesdf not installed')

class RobotModel:
    def __init__(self, urdf_path, n_surface_points=256, device='cpu', with_arm=True):
        self.device = device
        self.robot = Robot.from_xml_file(urdf_path)
        
        tactile_points = json.load(open(os.path.join(os.path.dirname(os.path.dirname(urdf_path)), 'addons/ur5_leap/tactile_points.json')))
        contact_points = json.load(open(os.path.join(os.path.dirname(os.path.dirname(urdf_path)), 'addons/ur5_leap/contact_points.json')))
        with open(os.path.join(os.path.dirname(os.path.dirname(urdf_path)), 'curobo/ur5_leap.yml'), 'r') as f:
            curobo_config = yaml.load(f, Loader=yaml.FullLoader)
        collision_spheres = curobo_config['robot_cfg']['kinematics']['collision_spheres']
        collision_spheres = {k: [dic['center'] + [dic['radius']] for dic in v] for k, v in collision_spheres.items()}

        # build chain
        self.joint_names = []
        self.joints_parent = []
        self.joints_child = []
        self.joints_type = []
        self.joints_axis_K = []
        self.joints_rotation = []
        self.joints_translation = []
        for joint in self.robot.joints:
            self.joint_names.append(joint.name)
            self.joints_parent.append(joint.parent)
            self.joints_child.append(joint.child)
            self.joints_type.append(joint.type)
            self.joints_axis_K.append(torch.tensor(joint.axis, dtype=torch.float, device=device).reshape(3, 1).expand(3, 3).cross(torch.eye(3, dtype=torch.float, device=device)) if joint.type == 'revolute' else None)
            self.joints_translation.append(torch.tensor(getattr(joint.origin, 'xyz', [0, 0, 0]), dtype=torch.float, device=device))
            self.joints_rotation.append(torch.tensor(transforms3d.euler.euler2mat(*getattr(joint.origin, 'rpy', [0, 0, 0]), 'sxyz'), dtype=torch.float, device=device))
        self.n_dofs = len([joint_type for joint_type in self.joints_type if joint_type != 'fixed'])
        if with_arm:
            self.joint_order = OrderedDict([
                ('shoulder_pan_joint', 0), ('shoulder_lift_joint', 1), ('elbow_joint', 2), ('wrist_1_joint', 3), ('wrist_2_joint', 4), ('wrist_3_joint', 5), ('ee_fixed_joint', -1), 
                ('j1', 6), ('j0', 7), ('j2', 8), ('j3', 9), 
                ('j12', 10), ('j13', 11), ('j14', 12), ('j15', 13), 
                ('j5', 14), ('j4', 15), ('j6', 16), ('j7', 17), 
                ('j9', 18), ('j8', 19), ('j10', 20), ('j11', 21)
            ])
        else:
            self.joint_order = OrderedDict([
                ('j1', 0), ('j0', 1), ('j2', 2), ('j3', 3),
                ('j12', 4), ('j13', 5), ('j14', 6), ('j15', 7),
                ('j5', 8), ('j4', 9), ('j6', 10), ('j7', 11),
                ('j9', 12), ('j8', 13), ('j10', 14), ('j11', 15)
            ])

        # build meshes
        self.mesh = {}
        areas = {}
        for link in self.robot.links:
            if link.visual is None or link.collision is None:
                continue
            self.mesh[link.name] = {}
            self.mesh[link.name]['boxes'] = []
            # load collision meshes
            link_mesh = tm.Trimesh()
            for collision in link.collisions:
                if type(collision.geometry) == Box:
                    translation = torch.tensor(getattr(collision.origin, 'xyz', [0, 0, 0]), dtype=torch.float, device=device)
                    rotation = torch.tensor(transforms3d.euler.euler2mat(*getattr(collision.origin, 'rpy', [0, 0, 0])), dtype=torch.float, device=device)
                    link_mesh += tm.primitives.Box(extents=collision.geometry.size).apply_transform(transforms3d.affines.compose(T=translation.cpu().numpy(), R=rotation.cpu().numpy(), Z=[1, 1, 1]))
                    size = torch.tensor(collision.geometry.size, dtype=torch.float, device=device) / 2
                    self.mesh[link.name]['boxes'].append({
                        'translation': translation, 
                        'rotation': rotation, 
                        'size': size, 
                    })
                elif type(collision.geometry) == Mesh:
                    filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(urdf_path))), collision.geometry.filename[10:])
                    link_mesh += tm.load_mesh(filename)
            vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            self.mesh[link.name].update({
                'collision_vertices': vertices, 
                'collision_faces': faces, 
            })
            # load visual mesh
            visual = link.visual
            filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(urdf_path))), visual.geometry.filename[10:])
            link_mesh = tm.load(filename, force='mesh')
            vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            translation = torch.tensor(getattr(visual.origin, 'xyz', [0, 0, 0]), dtype=torch.float, device=device)
            rotation = torch.tensor(transforms3d.euler.euler2mat(*getattr(visual.origin, 'rpy', [0, 0, 0])), dtype=torch.float, device=device)
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update({
                'visual_vertices': vertices, 
                'visual_faces': faces, 
            })
            # calculate visual mesh area
            areas[link.name] = link_mesh.area.item()
            # if link.name not in ['wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'hand_base_link', 'mcp_joint', 'pip', 'dip', 'fingertip', 'mcp_joint_2', 'pip_2', 'dip_2', 'fingertip_2', 'mcp_joint_3', 'pip_3', 'dip_3', 'fingertip_3', 'pip_4', 'thumb_pip', 'thumb_dip', 'thumb_fingertip']:
            #     areas[link.name] = 0
            # load tactile points
            self.mesh[link.name].update({
                'tactile_points': torch.tensor(tactile_points[link.name], dtype=torch.float, device=device), 
            })
            # load contact candidates
            self.mesh[link.name].update({
                'contact_candidates': torch.tensor(contact_points[link.name], dtype=torch.float, device=device) if link.name in contact_points else torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
            })
            self.mesh[link.name].update({
                'collision_spheres': torch.tensor(collision_spheres[link.name], dtype=torch.float, device=device) if link.name in collision_spheres else torch.tensor([], dtype=torch.float, device=device).reshape(0, 4)
            })

        # set joint limits
        self.joints_lower = torch.tensor([joint.limit.lower for joint in self.robot.joints if joint.joint_type == 'revolute'], dtype=torch.float, device=device)
        self.joints_upper = torch.tensor([joint.limit.upper for joint in self.robot.joints if joint.joint_type == 'revolute'], dtype=torch.float, device=device)
        index = [0] * self.n_dofs
        for i, joint_name in enumerate([joint.name for joint in self.robot.joints if joint.joint_type == 'revolute']):
            index[self.joint_order[joint_name]] = i
        self.joints_lower = self.joints_lower[index]
        self.joints_upper = self.joints_upper[index]
        
        # sample surface points
        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        num_samples['hand_base_link'] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]['surface_points'] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['visual_vertices'].unsqueeze(0), self.mesh[link_name]['visual_faces'].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]['surface_points'] = surface_points
        self.fingertip_bias = {k: torch.tensor(v, dtype=torch.float, device=device) for k, v in FINGERTIP_BIAS.items()}
        self.fingertip_normal = {k: torch.tensor(v, dtype=torch.float, device=device) for k, v in FINGERTIP_NORMAL.items()}

        # indexing
        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))
        self.surface_points_link_indices = torch.cat([self.link_name_to_link_index[link_name] * torch.ones(self.mesh[link_name]['surface_points'].shape[0], dtype=torch.long, device=device) for link_name in self.mesh])

        # build collision mask
        self.adjacency_mask = torch.zeros([len(self.mesh), len(self.mesh)], dtype=torch.bool, device=device)
        for joint in self.robot.joints:
            if joint.parent in self.mesh and joint.child in self.mesh:
                parent_id = self.link_name_to_link_index[joint.parent]
                child_id = self.link_name_to_link_index[joint.child]
                self.adjacency_mask[parent_id, child_id] = True
                self.adjacency_mask[child_id, parent_id] = True
        self.adjacency_mask[self.link_name_to_link_index['hand_base_link'], self.link_name_to_link_index['thumb_pip']] = True
        self.adjacency_mask[self.link_name_to_link_index['thumb_pip'], self.link_name_to_link_index['hand_base_link']] = True
        self.adjacency_mask[self.link_name_to_link_index['hand_base_link'], self.link_name_to_link_index['thumb_dip']] = True
        self.adjacency_mask[self.link_name_to_link_index['thumb_dip'], self.link_name_to_link_index['hand_base_link']] = True
        self.adjacency_mask[self.link_name_to_link_index['mcp_joint'], self.link_name_to_link_index['dip']] = True
        self.adjacency_mask[self.link_name_to_link_index['dip'], self.link_name_to_link_index['mcp_joint']] = True
        self.adjacency_mask[self.link_name_to_link_index['mcp_joint_2'], self.link_name_to_link_index['dip_2']] = True
        self.adjacency_mask[self.link_name_to_link_index['dip_2'], self.link_name_to_link_index['mcp_joint_2']] = True
        self.adjacency_mask[self.link_name_to_link_index['mcp_joint_3'], self.link_name_to_link_index['dip_3']] = True
        self.adjacency_mask[self.link_name_to_link_index['dip_3'], self.link_name_to_link_index['mcp_joint_3']] = True
        
        # build collision mask
        self.adjacency_mask = torch.zeros([len(self.mesh), len(self.mesh)], dtype=torch.bool, device=device)
        for joint in self.robot.joints:
            if joint.parent in self.mesh and joint.child in self.mesh:
                parent_id = self.link_name_to_link_index[joint.parent]
                child_id = self.link_name_to_link_index[joint.child]
                self.adjacency_mask[parent_id, child_id] = True
                self.adjacency_mask[child_id, parent_id] = True
        self.adjacency_mask[self.link_name_to_link_index['hand_base_link'], self.link_name_to_link_index['thumb_pip']] = True
        self.adjacency_mask[self.link_name_to_link_index['thumb_pip'], self.link_name_to_link_index['hand_base_link']] = True
        self.adjacency_mask[self.link_name_to_link_index['hand_base_link'], self.link_name_to_link_index['thumb_dip']] = True
        self.adjacency_mask[self.link_name_to_link_index['thumb_dip'], self.link_name_to_link_index['hand_base_link']] = True
        self.adjacency_mask[self.link_name_to_link_index['mcp_joint'], self.link_name_to_link_index['dip']] = True
        self.adjacency_mask[self.link_name_to_link_index['dip'], self.link_name_to_link_index['mcp_joint']] = True
        self.adjacency_mask[self.link_name_to_link_index['mcp_joint_2'], self.link_name_to_link_index['dip_2']] = True
        self.adjacency_mask[self.link_name_to_link_index['dip_2'], self.link_name_to_link_index['mcp_joint_2']] = True
        self.adjacency_mask[self.link_name_to_link_index['mcp_joint_3'], self.link_name_to_link_index['dip_3']] = True
        self.adjacency_mask[self.link_name_to_link_index['dip_3'], self.link_name_to_link_index['mcp_joint_3']] = True
        
        self.hand_pose = None
        self.global_translation = None
        self.global_rotation = None
        self.local_translations = None
        self.local_rotations = None

    def set_parameters(self, hand_pose):
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(self.hand_pose[:, 3:9])
        batch_size = len(self.hand_pose)
        self.local_translations = {}
        self.local_rotations = {}
        self.local_translations[self.joints_parent[0]] = torch.zeros([batch_size, 3], dtype=torch.float, device=self.device)
        self.local_rotations[self.joints_parent[0]] = torch.eye(3, dtype=torch.float, device=self.device).expand(batch_size, 3, 3).contiguous()
        for joint_name, j in self.joint_order.items():
            i = self.joint_names.index(joint_name)
            child_name = self.joints_child[i]
            parent_name = self.joints_parent[i]
            translations = self.local_rotations[parent_name] @ self.joints_translation[i] + self.local_translations[parent_name]
            rotations = self.local_rotations[parent_name] @ self.joints_rotation[i]
            if self.joints_type[i] == 'revolute':
                thetas = self.hand_pose[:, 9 + j].view(batch_size, 1, 1)
                K = self.joints_axis_K[i]
                joint_rotations = torch.eye(3, dtype=torch.float, device=self.device) + torch.sin(thetas) * K + (1 - torch.cos(thetas)) * (K @ K)
                rotations = rotations @ joint_rotations
            self.local_translations[child_name] = translations
            self.local_rotations[child_name] = rotations
    
    def set_parameters_simple(self, qpos):
        add = torch.zeros_like(qpos[:, :9])
        add[:, 3] = 1
        add[:, 7] = 1
        self.set_parameters(torch.cat([add, qpos], dim=-1))
    
    def cal_distance(self, x, dilation_pen=0):
        # x: (total_batch_size, num_samples, 3)
        # 单独考虑每个link
        # 先把x变换到link的局部坐标系里面
        # 再变换到collision primitive的中心坐标系里
        # 得到x_local: (total_batch_size, num_samples, 3)
        # 然后计算dis，按照内外取符号，内部是正号
        # 最后的dis就是所有link的dis的最大值
        # collsion primitive只有sphere和box，都是用解析sdf
        # TODO: forward 15% backward 20%
        dis = []
        # x = transform_points_inverse(x, self.global_translation, self.global_rotation)
        # equivalent to:
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            # x_local = transform_points_inverse(x, self.local_translations[link_name], self.local_rotations[link_name])
            # equivalent to:
            x_local = (x - self.local_translations[link_name].unsqueeze(1)) @ self.local_rotations[link_name]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            for box in self.mesh[link_name]['boxes']:
                # x_box = fixed_transform_points_inverse(x_local, box['translation'], box['rotation'])
                # equivalent to:
                x_box = (x_local - box['translation']) @ box['rotation']
                size = box['size'] + dilation_pen
                q = torch.abs(x_box) - size
                dis_local = torch.max(q, torch.zeros_like(q)).norm(p=2, dim=-1) + torch.min(torch.max(q, dim=-1)[0], torch.zeros_like(q).max(dim=-1)[0])
                dis_local = -dis_local
                dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def get_collision_spheres(self):
        points = []
        radius = []
        for link_name in self.mesh:
            if len(self.mesh[link_name]['collision_spheres']) > 0:
                points.append(self.mesh[link_name]['collision_spheres'][..., :3] @ self.local_rotations[link_name].transpose(1, 2) + self.local_translations[link_name].unsqueeze(1))
                radius.append(self.mesh[link_name]['collision_spheres'][..., 3])
        points = torch.cat(points, dim=-2).to(self.device)
        radius = torch.cat(radius, dim=-1).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points, radius
    
    def cal_self_distance(self, dilation_spen=0):
        # TODO: forward 8% backward 10%
        # get surface points
        x = []
        for link_name in self.mesh:
            x.append(self.mesh[link_name]['surface_points'] @ self.local_rotations[link_name].transpose(1, 2) + self.local_translations[link_name].unsqueeze(1))
        x = torch.cat(x, dim=-2).to(self.device)  # (total_batch_size, n_surface_points, 3)
        # x = x.detach()  # or else gradient will backprop through x to hand_pose
        if len(x.shape) == 2:
            x = x.expand(1, x.shape[0], x.shape[1])
        # cal distance
        dis = []
        for link_name in self.mesh:
            x_local = transform_points_inverse(x, self.local_translations[link_name], self.local_rotations[link_name])
            # equivalent to:
            # x_local = (x - self.local_translations[link_name].unsqueeze(1)) @ self.local_rotations[link_name]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * n_surface_points, 3)
            for box in self.mesh[link_name]['boxes']:
                x_box = fixed_transform_points_inverse(x_local, box['translation'], box['rotation'])
                # equivalent to:
                # x_local = (x_local - self.mesh[link_name]['translation']) @ self.mesh[link_name]['rotation']
                size = box['size'] + dilation_spen
                dis_local, dis_signs, _ = box_sdf(x_box, size)
                dis_local = (dis_local + 1e-8).sqrt()
                dis_local = torch.where(dis_signs, -dis_local, dis_local)
                dis_local = dis_local.reshape(x.shape[0], x.shape[1])  # (total_batch_size, n_surface_points)
                is_adjacent = self.adjacency_mask[self.link_name_to_link_index[link_name], self.surface_points_link_indices]  # (n_surface_points,)
                dis_local[:, is_adjacent | (self.link_name_to_link_index[link_name] == self.surface_points_link_indices)] = -float('inf')
                dis.append(dis_local)
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def self_penetration(self, dilation_spen=0):
        dis = self.cal_self_distance(dilation_spen=dilation_spen)
        dis[dis <= 0] = 0
        E_spen = dis.sum(-1)
        return E_spen
    
    def cal_loss_tpen(self, p, dilation_tpen=0.):
        collision_vertices = self.get_collision_vertices()
        dis = (p[:, :3].unsqueeze(1) * collision_vertices).sum(2) + p[:, 3].unsqueeze(1) - dilation_tpen
        dis[dis > 0] = 0
        loss_tpen = -dis.sum() / len(collision_vertices)
        return loss_tpen
    
    def get_fingertips(self):
        result = []
        for link_name in self.mesh:
            if not 'fingertip' in link_name:
                continue
            local_translation = torch.einsum('nab,b->na', self.local_rotations[link_name], self.fingertip_bias[link_name]) + self.local_translations[link_name]
            pos = torch.einsum('nab,nb->na', self.global_rotation, local_translation) + self.global_translation
            result.append(pos)
        return torch.stack(result, dim=-2)   
    
    def get_fingertip_normals(self):
        result = []
        for link_name in self.mesh:
            if not 'fingertip' in link_name:
                continue
            local_rotation = self.local_rotations[link_name]
            normal = torch.einsum('nab,b->na', local_rotation, self.fingertip_normal[link_name])
            normal = torch.einsum('nab,nb->na', self.global_rotation, normal)
            result.append(normal)
        return torch.stack(result, dim=-2)
        
    def get_tactile_points(self):
        points = []
        for link_name in self.mesh:
            if len(self.mesh[link_name]['tactile_points']) > 0:
                points.append(self.mesh[link_name]['tactile_points'] @ self.local_rotations[link_name].transpose(1, 2) + self.local_translations[link_name].unsqueeze(1))
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points
    
    def get_surface_points(self, nobase=False):
        points = []
        for link_name in self.mesh:
            if nobase and link_name in ['base_link', 'shoulder_link']:
                continue
            points.append(self.mesh[link_name]['surface_points'] @ self.local_rotations[link_name].transpose(1, 2) + self.local_translations[link_name].unsqueeze(1))
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_collision_vertices(self, link_names=None):
        if link_names is None:
            link_names = self.mesh.keys()
        points = []
        for link_name in link_names:
            points.append(self.mesh[link_name]['collision_vertices'] @ self.local_rotations[link_name].transpose(1, 2) + self.local_translations[link_name].unsqueeze(1))
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points
    
    def get_contact_candidates(self):
        points = []
        for link_name in self.mesh:
            if len(self.mesh[link_name]['contact_candidates']) > 0:
                points.append(self.mesh[link_name]['contact_candidates'] @ self.local_rotations[link_name].transpose(1, 2) + self.local_translations[link_name].unsqueeze(1))
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points
    
    def get_squeeze_pose(self, target_distance_thumb, target_distance_others, grad_move_hand, grad_move_arm, num_iters, use_center=False):
        # get fingertips
        fingertips = self.get_fingertips()
        fingertip_normals = self.get_fingertip_normals()
        thumb = fingertips[:, 3]
        others = fingertips[:, :3]
        if use_center:
            center = others.mean(dim=1)
            direction_thumb = (center - thumb).unsqueeze(1)
            direction_thumb = direction_thumb / direction_thumb.norm(dim=-1, keepdim=True)
            thumb_target = thumb + direction_thumb * target_distance_thumb
            direction_others = (thumb.unsqueeze(1) - others)
            direction_others = direction_others / direction_others.norm(dim=-1, keepdim=True)
            others_target = others + direction_others * target_distance_others
            fingertips_target =  torch.cat([others_target, thumb_target], dim=1).detach()
        else:
            thumb_target = thumb + fingertip_normals[:, 3] * target_distance_thumb
            others_target = others + fingertip_normals[:, :3] * target_distance_others
            fingertips_target =  torch.cat([others_target, thumb_target.unsqueeze(1)], dim=1).detach()
        # squeeze with gradient descent
        squeeze_pose = self.hand_pose.clone()
        squeeze_pose.requires_grad_()
        if squeeze_pose.grad is not None:
            squeeze_pose.grad.zero_()
        losses = []
        for step in range(num_iters):
            self.set_parameters(squeeze_pose)
            fingertips = self.get_fingertips()
            loss = (fingertips_target - fingertips).square().sum()
            loss.backward()
            losses.append(loss.item())
            with torch.no_grad():
                squeeze_pose[:, 15:] = squeeze_pose[:, 15:] - self.hand_pose.grad[:, 15:] * grad_move_hand
                squeeze_pose[:, 9:15] = squeeze_pose[:, 9:15] - self.hand_pose.grad[:, 9:15] * grad_move_arm
                squeeze_pose.grad.zero_()
        return squeeze_pose, losses, fingertips_target
        
    
    def get_plotly_data(self, i, opacity=0.5, color='lightblue', visual=True, pose=None):
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        vmin = 10000
        for link_name in self.mesh:
            v = self.mesh[link_name]['collision_vertices'] if not visual else self.mesh[link_name]['visual_vertices']
            v = v @ self.local_rotations[link_name][i].T + self.local_translations[link_name][i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = (self.mesh[link_name]['collision_faces'] if not visual else self.mesh[link_name]['visual_faces']).detach().cpu()
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            vmin = min(vmin, v[:, 2].min())
            data.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], text=[link_name] * len(v), color=color, opacity=opacity, hovertemplate='%{text}'))
        # print(vmin)
        return data
