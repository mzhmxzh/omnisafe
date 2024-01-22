import torch
from einops import repeat, rearrange
# from pointnet2_ops.pointnet2_utils import furthest_point_sample


def fps(pc, num):
    if pc.shape[1] == num:
        return pc
    return torch.gather(pc, dim=1, index=repeat(furthest_point_sample(pc, num).long(), 'b n -> b n d', d=pc.shape[-1]))

def joint_fps(robot_pc, obj_pc, n_robot_points, n_obj_points):
    robot_fps_pc = fps(robot_pc, n_robot_points)
    obj_fps_pc = fps(obj_pc, n_obj_points)
    return combine_observation(robot_fps_pc, obj_fps_pc)

def select_pc(robot_pc, obj_pc, n_robot_points, n_obj_points, pc_type='joint'):
    if pc_type == 'robot':
        return fps(robot_pc, n_robot_points)
    elif pc_type == 'obj':
        return fps(obj_pc, n_obj_points)
    elif pc_type == 'joint':
        return joint_fps(robot_pc, obj_pc, n_robot_points, n_obj_points)
    else:
        raise NotImplementedError

def get_target_pc(robot_model, state, full_obj_pc, n_robot_points, n_obj_points, pc_type):
    batch_size = state.shape[0]
    device = state.device
    robot_pose = torch.cat([
        repeat(torch.tensor([0, 0, 0], dtype=torch.float, device=device), 'd -> b d', b=batch_size), 
        repeat(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device=device), 'd -> b d', b=batch_size),
        state[:, 12:], 
    ], dim=-1)
    robot_model.set_parameters(robot_pose)
    full_obj_pc = torch.einsum('b x y, b n y -> b n x', state[:, 3:12].reshape(-1, 3, 3), full_obj_pc) + state[:, None, :3]
    #full_robot_pc, tac_pc = robot_model.get_batch_pc(visual=True, tactile=True, obj_pc=full_obj_pc)
    full_robot_pc = robot_model.get_surface_points()
    tac_pc = robot_model.get_tactile_points()
    return select_pc(full_robot_pc, full_obj_pc.contiguous(), n_robot_points, n_obj_points, pc_type), tac_pc

def combine_observation(robot_pc, obj_pc, has_obj_pc=None):
    robot_seg_pc = torch.cat([robot_pc, torch.zeros_like(robot_pc[..., [0]])], dim=-1)
    obj_seg_pc = torch.cat([obj_pc, torch.ones_like(obj_pc[..., [0]]) * (1 if has_obj_pc is None else has_obj_pc[:, None, None])], dim=-1)
    return torch.cat([robot_seg_pc, obj_seg_pc], dim=1)

def update_seg(pc, tac_pc, with_tac=1):
    new_pc = torch.zeros((pc.shape[0], pc.shape[1]+tac_pc.shape[1], 3+3+1), device=pc.device)
    new_pc[:, :pc.shape[1], :3] = pc[..., :3]
    new_pc[:, :pc.shape[1], 3] = pc[..., 3]
    new_pc[:, :pc.shape[1], 4] = 1-pc[..., 3]
    new_pc[:, pc.shape[1]:, :3] = tac_pc[..., :3]
    new_pc[:, pc.shape[1]:, 5] = 1
    #new_pc[:, pc.shape[1]:, 6] = tac_pc[..., 3] * with_tac
    new_pc[:, pc.shape[1]:, 6] = 1
    return new_pc

def sym_approx_emd(loss_fn, pc1, pc2):
    pc_left = torch.cat([pc1, pc2], dim=0).contiguous()
    pc_right = torch.cat([pc2, pc1], dim=0).contiguous()
    return loss_fn(pc_left, pc_right) * 2