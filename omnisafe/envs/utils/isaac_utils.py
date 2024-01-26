import numpy as np
import os
import torch
from isaacgym import gymapi, gymtorch
# from torchsdf.sdf import compute_sdf
from utils.point_cloud import update_seg, get_target_pc
from utils.robot_info import ROBOT_BASE_HEIGHT, TABLE_HEIGHT
from isaacgym.torch_utils import quat_apply

joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 
    'j1', 'j0', 'j2', 'j3',
    'j12', 'j13', 'j14', 'j15',
    'j5', 'j4', 'j6', 'j7',
    'j9', 'j8', 'j10', 'j11',
]
joint_names_isaac = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 
    'j1', 'j0', 'j2', 'j3',
    'j12', 'j13', 'j14', 'j15',
    'j5', 'j4', 'j6', 'j7',
    'j9', 'j8', 'j10', 'j11',
]
joint_stiffness = dict(
    shoulder_pan_joint=10000, shoulder_lift_joint=10000, elbow_joint=10000, wrist_1_joint=10000, wrist_2_joint=10000, wrist_3_joint=10000, 
    j0=10000, j1=10000, j2=10000, j3=10000,
    j4=10000, j5=10000, j6=10000, j7=10000,
    j8=10000, j9=10000, j10=10000, j11=10000,
    j12=10000, j13=10000, j14=10000, j15=10000,
)
joint_damping = dict(
    shoulder_pan_joint=250, shoulder_lift_joint=250, elbow_joint=250, wrist_1_joint=250, wrist_2_joint=250, wrist_3_joint=250, 
    j0=250, j1=250, j2=250, j3=250,
    j4=250, j5=250, j6=250, j7=250,
    j8=250, j9=250, j10=250, j11=250,
    j12=250, j13=250, j14=250, j15=250,
)

joint_effort = dict(
    j0=10.95, j1=10.95, j2=10.95, j3=10.95,
    j4=10.95, j5=10.95, j6=10.95, j7=10.95,
    j8=10.95, j9=10.95, j10=10.95, j11=10.95,
    j12=10.95, j13=10.95, j14=10.95, j15=10.95,
)
joint_drive_mode = dict(
    shoulder_pan_joint=gymapi.DOF_MODE_POS, shoulder_lift_joint=gymapi.DOF_MODE_POS, elbow_joint=gymapi.DOF_MODE_POS, wrist_1_joint=gymapi.DOF_MODE_POS, wrist_2_joint=gymapi.DOF_MODE_POS, wrist_3_joint=gymapi.DOF_MODE_POS, 
    j0=gymapi.DOF_MODE_POS, j1=gymapi.DOF_MODE_POS, j2=gymapi.DOF_MODE_POS, j3=gymapi.DOF_MODE_POS,
    j4=gymapi.DOF_MODE_POS, j5=gymapi.DOF_MODE_POS, j6=gymapi.DOF_MODE_POS, j7=gymapi.DOF_MODE_POS,
    j8=gymapi.DOF_MODE_POS, j9=gymapi.DOF_MODE_POS, j10=gymapi.DOF_MODE_POS, j11=gymapi.DOF_MODE_POS,
    j12=gymapi.DOF_MODE_POS, j13=gymapi.DOF_MODE_POS, j14=gymapi.DOF_MODE_POS, j15=gymapi.DOF_MODE_POS,
)

def get_point_on_link(link_state, point):
    link_trans = link_state[:, :3]
    link_quat = link_state[:, 3:7]
    if len(point.shape) == 1:
        point = point.expand(len(link_state), 3)
    return link_trans + quat_apply(link_quat, point)

def points_in_contact(obj_points, pos, rot, size, thres):
    obj_points = torch.einsum('nab,nka->nkb', rot, obj_points - pos[:, None])
    near_points = obj_points.clone()
    near_points = torch.where(near_points > size[None, None, :]/2, size[None, None, :]/2, near_points)
    near_points = torch.where(near_points < -size[None, None, :]/2, -size[None, None, :]/2, near_points)
    dis = torch.norm(obj_points - near_points, dim=-1)
    min_dis = torch.min(dis, dim=-1)[0]
    return (min_dis < thres).float()

def get_sim_params(config):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./config.env_hz
    sim_params.substeps = config.sim.substeps
    sim_params.num_client_threads = 0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 16
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.00625
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.2
    sim_params.physx.max_depenetration_velocity = 5.0
    sim_params.physx.default_buffer_size_multiplier = 8.0
    sim_params.use_gpu_pipeline = False if config.device=='cpu' else True
    sim_params.physx.num_threads = 4
    sim_params.physx.num_subscenes = 4
    sim_params.physx.use_gpu = False if config.device=='cpu' else True
    sim_params.physx.max_gpu_contact_pairs = 50 * 1024 * 1024
    sim_params.physx.contact_collection = gymapi.ContactCollection(1)
    return sim_params

def update_sim_params(sim_params, config):
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.gravity.z = sim_params.gravity.z * (np.random.rand(1)[0]* (config.gravity_range[1] - config.gravity_range[0]) + config.gravity_range[0])

def get_plane_params():
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    return plane_params

def get_robot_asset_options():
    robot_asset_options = gymapi.AssetOptions()
    robot_asset_options.fix_base_link = True
    robot_asset_options.collapse_fixed_joints = False
    # robot_asset_options.flip_visual_attachments = False
    robot_asset_options.disable_gravity = True
    robot_asset_options.thickness = 0.0
    robot_asset_options.density = 1000.0
    robot_asset_options.armature = 0.01
    robot_asset_options.angular_damping = 5.0
    robot_asset_options.linear_damping = 1.0
    robot_asset_options.max_linear_velocity = 1.0
    robot_asset_options.max_angular_velocity = 2 * np.pi
    robot_asset_options.enable_gyroscopic_forces = True
    robot_asset_options.use_physx_armature = True
    robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    return robot_asset_options

def get_table_asset_options():
    table_options = gymapi.AssetOptions()
    table_options.flip_visual_attachments = False  # default = False
    table_options.fix_base_link = True
    table_options.thickness = 0.0  # default = 0.02
    table_options.density = 1000.0  # default = 1000.0
    table_options.armature = 0.0  # default = 0.0
    table_options.use_physx_armature = True
    table_options.linear_damping = 0.0  # default = 0.0
    table_options.max_linear_velocity = 1000.0  # default = 1000.0
    table_options.angular_damping = 0.0  # default = 0.5
    table_options.max_angular_velocity = 64.0  # default = 64.0
    table_options.disable_gravity = False
    table_options.enable_gyroscopic_forces = True
    table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    table_options.use_mesh_materials = False
    return table_options

def get_object_asset_options():
    object_asset_options = gymapi.AssetOptions()
    object_asset_options.flip_visual_attachments = False
    object_asset_options.fix_base_link = False
    object_asset_options.thickness = 0.0  # default = 0.02
    object_asset_options.density = 1000.0
    object_asset_options.armature = 0.0  # default = 0.0
    object_asset_options.use_physx_armature = True
    object_asset_options.linear_damping = 0.5  # default = 0.0
    object_asset_options.max_linear_velocity = 1000.0  # default = 1000.0
    object_asset_options.angular_damping = 0.5  # default = 0.5
    object_asset_options.max_angular_velocity = 64.0  # default = 64.0
    object_asset_options.disable_gravity = False
    object_asset_options.enable_gyroscopic_forces = True
    object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    object_asset_options.use_mesh_materials = False
    object_asset_options.vhacd_enabled = True
    object_asset_options.vhacd_params = gymapi.VhacdParams()
    object_asset_options.vhacd_params.resolution = 300000
    return object_asset_options

def create_table_actor(gym, env, sim, table_asset, table_dims, all_indices=[]):
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
    table_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
    table_actor_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, -1)
    table_texture_files = "/home/jialiangzhang/Workspace/tacdexgrasp/assets/textures/texture_stone_stone_texture_0.jpg"
    if os.path.exists(table_texture_files):
        table_texture_handle = gym.create_texture_from_file(sim, table_texture_files)   
        gym.set_rigid_body_texture(env, table_actor_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
    table_shape_props = gym.get_actor_rigid_shape_properties(env, table_actor_handle)
    table_shape_props[0].friction = 1
    gym.set_actor_rigid_shape_properties(env, table_actor_handle, table_shape_props)
    all_indices.append(gym.get_actor_index(env, table_actor_handle, gymapi.DOMAIN_SIM))

def create_robot_actor(gym, env, robot_asset, table_dims, robot_indices=[], all_indices=[], friction=1, p=None, d=None):
    allegro_hand_start_pose = gymapi.Transform()
    allegro_hand_start_pose.p = gymapi.Vec3(0, 0, table_dims.z + ROBOT_BASE_HEIGHT)
    allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
    robot_actor_handle = gym.create_actor(env, robot_asset, allegro_hand_start_pose, "robot", 0, -1, 3)
    robot_props = gym.get_actor_dof_properties(env, robot_actor_handle)
    for joint_name in joint_drive_mode:
        robot_props['driveMode'][joint_names_isaac.index(joint_name)] = joint_drive_mode[joint_name]
    for joint_name in joint_stiffness:
        robot_props['stiffness'][joint_names_isaac.index(joint_name)] = joint_stiffness[joint_name] if p is None else p
    for joint_name in joint_damping:
        robot_props['damping'][joint_names_isaac.index(joint_name)] = joint_damping[joint_name] if d is None else d
    for joint_name in joint_effort:
        robot_props['effort'][joint_names_isaac.index(joint_name)] = joint_effort[joint_name]
    lower_limits = []
    upper_limits = []
    for name in joint_names_isaac:
        lower_limits.append(robot_props['lower'][joint_names_isaac.index(name)])
        upper_limits.append(robot_props['upper'][joint_names_isaac.index(name)])
    gym.set_actor_dof_properties(env, robot_actor_handle, robot_props)
    robot_shape_props = gym.get_actor_rigid_shape_properties(env, robot_actor_handle)
    for j in range(len(robot_shape_props)):
        robot_shape_props[j].friction = friction
        assert robot_shape_props[j].restitution == 0.0, robot_shape_props[j].restitution
    gym.set_actor_rigid_shape_properties(env, robot_actor_handle, robot_shape_props)
    robot_indices.append(gym.get_actor_index(env, robot_actor_handle, gymapi.DOMAIN_SIM))
    all_indices.append(gym.get_actor_index(env, robot_actor_handle, gymapi.DOMAIN_SIM))
    return lower_limits, upper_limits, robot_actor_handle, robot_props
 
def init_waiting_pose(num_obj_per_env, table_dims, env_spacing, object_rise, width=0.1, a=15, b=15):
    assert num_obj_per_env <= a * b + 1
    half_table = max(table_dims.x, table_dims.y) * 0.5
    effective_env_spacing_row = half_table + width * (a+1)
    effective_env_spacing_col = width * (b+1) + half_table
    assert env_spacing > 1.2*max(effective_env_spacing_row, effective_env_spacing_col)
    waiting_pose = np.zeros((a, b, 3))
    waiting_pose[:, :, 0] = (np.arange(a) * width + half_table + width)[:, None]
    waiting_pose[:, :, 1] = np.arange(b) * width + half_table + width
    waiting_pose[:, :, 2] = object_rise
    waiting_pose = waiting_pose.reshape(-1, 3)

    init_object_waiting_pose = []
    for i in range(a * b):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*list(waiting_pose[i, 0:3]))
        init_object_waiting_pose.append(pose)
    return init_object_waiting_pose

def get_camera_params():
    camera_props = gymapi.CameraProperties()
    camera_props.width = 320
    camera_props.height = 288
    camera_props.horizontal_fov = 75
    camera_props.enable_tensors = True
    return camera_props

def load_cameras(gym, sim, env_ptr, device, env_id, camera_props, camera_eye, camera_lookat, camera_depth_tensor_list, camera_rgb_tensor_list, camera_seg_tensor_list, camera_vinv_mat_list, camera_proj_mat_list, env_origin):

    camera_handles = []
    depth_tensors = []
    rgb_tensors = []
    seg_tensors = []
    vinv_mats = []
    proj_mats = []

    origin = gym.get_env_origin(env_ptr)
    env_origin[env_id, 0] = origin.x
    env_origin[env_id, 1] = origin.y
    env_origin[env_id, 2] = origin.z

    camera_handle = gym.create_camera_sensor(env_ptr, camera_props)


    gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(*camera_eye.cpu().numpy()), gymapi.Vec3(*camera_lookat.cpu().numpy()))

    raw_depth_tensor = gym.get_camera_image_gpu_tensor(sim, env_ptr, camera_handle,
                                                            gymapi.IMAGE_DEPTH)
    depth_tensor = gymtorch.wrap_tensor(raw_depth_tensor)
    depth_tensors.append(depth_tensor)

    raw_rgb_tensor = gym.get_camera_image_gpu_tensor(sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
    rgb_tensor = gymtorch.wrap_tensor(raw_rgb_tensor)
    rgb_tensors.append(rgb_tensor)

    raw_seg_tensor = gym.get_camera_image_gpu_tensor(sim, env_ptr, camera_handle,
                                                            gymapi.IMAGE_SEGMENTATION)
    seg_tensor = gymtorch.wrap_tensor(raw_seg_tensor)
    seg_tensors.append(seg_tensor)

    vinv_mat = torch.inverse(
        (torch.tensor(gym.get_camera_view_matrix(sim, env_ptr, camera_handle), device=device))
    )
    vinv_mats.append(vinv_mat)

    proj_mat = torch.tensor(gym.get_camera_proj_matrix(sim, env_ptr, camera_handle), device=device)
    proj_mats.append(proj_mat)

    camera_handles.append(camera_handle)

    camera_depth_tensor_list+=depth_tensors
    camera_rgb_tensor_list+=rgb_tensors
    camera_seg_tensor_list+=seg_tensors
    camera_vinv_mat_list+=vinv_mats
    camera_proj_mat_list+=proj_mats

    return

def collect_pointclouds_legacy(gym, sim, robot_model, robot_state, num_envs, t, camera_props, camera_u2, camera_v2, env_origin, camera_depth_tensor_list, camera_rgb_tensor_list, camera_seg_tensor_list, camera_vinv_mat_list, camera_proj_mat_list, device, num_cameras=1):

    robot_translation = torch.zeros([len(robot_state), 3], dtype=torch.float, device=robot_state.device)
    robot_translation[:, 2] += 0.6
    robot_rotation = torch.eye(3, dtype=torch.float, device=robot_state.device).expand(len(robot_state), 3, 3)
    robot_pose = torch.cat([
        robot_translation,
        robot_rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        robot_state[:, :22], 
    ], dim=-1)
    robot_model.set_parameters(robot_pose)
    robot_pc_fps = robot_model.get_surface_points()

    # TODO: i will create a config file later
    x_n_bar = -5
    x_p_bar = 5
    y_n_bar = -5
    y_p_bar = 5
    z_n_bar = 0.605
    z_p_bar = 5
    depth_bar = 5
    hand_sample_num = 1024
    obj_sample_num = 256
    num_pc_downsample = 1280
    num_pc_presample = 65536

    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    def split_first_dim(tensor, camera_num):
        return tensor.reshape(-1, camera_num, *tensor.shape[1:])

    depth_tensor = split_first_dim(torch.stack(camera_depth_tensor_list), num_cameras)
    rgb_tensor = split_first_dim(torch.stack(camera_rgb_tensor_list), num_cameras)
    seg_tensor = split_first_dim(torch.stack(camera_seg_tensor_list), num_cameras)
    vinv_mat = split_first_dim(torch.stack(camera_vinv_mat_list), num_cameras)
    proj_matrix = split_first_dim(torch.stack(camera_proj_mat_list), num_cameras)

    def visualize_sensors(env_id=0, t=0):
        COLOR3 =  torch.tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], device=device)
        rgb_img = rgb_tensor[..., :3].to(device)
        background = torch.zeros_like(rgb_img)
        background[..., 0] = 255
        seg_img = COLOR3[seg_tensor.long().to(device)].to(device)
        rgb_img = torch.where((seg_tensor==0).unsqueeze(-1).to(device), background, rgb_img)
        depth_img = depth_tensor.to(device)
        def save_img2(name, rgb_img, seg_img, dep_img):
            import cv2
            import os
            save_dir = 'tmp/vis/diffusion'
            os.makedirs(save_dir, exist_ok=True)
            # cv2.imwrite(osp.join(save_dir, f'{name}_depth.png'), dep_img)
            cv2.imwrite(os.path.join(save_dir, f'{name}_rgb.png'), rgb_img)
            # cv2.imwrite(osp.join(save_dir, f'{name}_seg.png'), seg_img)
            return
        rgb_img = rgb_tensor[env_id, 0, :, :, :3].cpu().numpy()
        seg_img = COLOR3[seg_tensor[env_id,0].long().to(device)].cpu().numpy()
        dep_img = np.zeros_like(rgb_img)
        dep_img[..., 2] = -100 * depth_img[env_id].cpu().numpy()
        save_img2(f'env_{env_id}_{t}_camera', rgb_img, seg_img, dep_img)
        return

    # print(step_counter)
    # print(root_state_tensor[hand_indices, 0:3])
    # pdb.set_trace()
    # for i in range(num_envs):
    visualize_sensors(0, t)

    point_list = []
    valid_list = []
    for i in range(num_cameras):
        # (num_envs, num_pts, 7) (num_envs, num_pts)
        point, valid = depth_image_to_point_cloud_GPU_batch(depth_tensor[:, i], rgb_tensor[:, i], seg_tensor[:, i],
                                                            vinv_mat[:, i], proj_matrix[:, i],
                                                            camera_u2, camera_v2, camera_props.width,
                                                            camera_props.height, depth_bar, device,
                                                            # z_p_bar, z_n_bar
                                                            )
        point_list.append(point)
        valid_list.append(valid)

        # print(f'camera {i}, ', valid.sum(dim=1))

    # (num_envs, 65536 * num_cameras, 7)
    points = torch.cat(point_list, dim=1)
    depth_mask = torch.cat(valid_list, dim=1)
    points[:, :, :3] -= env_origin.view(num_envs, 1, 3)
    # if headless:
    #     points[:, :, :3] -= env_origin.view(num_envs, 1, 3) * 2
    # else:
    #     points[:, :, :3] -= env_origin.view(num_envs, 1, 3)

    x_mask = (points[:, :, 0] > x_n_bar) * (points[:, :, 0] < x_p_bar)
    y_mask = (points[:, :, 1] > y_n_bar) * (points[:, :, 1] < y_p_bar)
    z_mask = (points[:, :, 2] > z_n_bar) * (points[:, :, 2] < z_p_bar)
    # (num_envs, 65536 * 3)
    valid = depth_mask * x_mask * y_mask * z_mask

    # (num_envs,)
    point_nums = valid.sum(dim=1)
    now = 0
    points_list = []
    # (num_valid_pts_total, 7)
    valid_points = points[valid]

    # presample, make num_pts equal for each env
    for env_id, point_num in enumerate(point_nums):
        if point_num == 0:
            print(f'env{env_id}_____point_num = 0_____')
            continue
        points_all = valid_points[now: now + point_num]
        random_ids = torch.randint(0, points_all.shape[0], (num_pc_presample,), device=device,
                                    dtype=torch.long)
        points_all_rnd = points_all[random_ids]
        points_list.append(points_all_rnd)
        now += point_num
    
    assert len(points_list) == num_envs, f'{num_envs - len(points_list)} envs have 0 point'
    # (num_envs, num_pc_presample)
    points_batch = torch.stack(points_list)

    assert hand_sample_num + obj_sample_num == num_pc_downsample

    first_indices = torch.zeros((num_envs, num_pc_presample), device=device, dtype=torch.bool)
    first_indices[:, 0] = True # in case there are no obj points
    hand_mask = (points_batch[:, :, -1] == 3)
    object_mask = torch.logical_or(first_indices, points_batch[:, :, -1] == 2)
    hand_pc = rand_select(points_batch, hand_mask, num_pc_presample)
    object_pc = rand_select(points_batch, object_mask, num_pc_presample)
    #hand_pc = points_batch.view(-1, 7)[hand_idx]
    #object_idx = torch.where(, idx, zeros)
    #object_pc = points_batch.view(-1, 7)[object_idx]
    hand_fps = sample_points(hand_pc, sample_num=hand_sample_num,
                                sample_method='furthest_batch', device=device)
    object_fps = sample_points(object_pc, sample_num=obj_sample_num,
                                sample_method='furthest_batch', device=device)
    points_fps = torch.cat([hand_fps, object_fps], dim=1)

    def visualize_pc(env_id, camera_id, mode='rgb', pc=points_fps):
        vis_points = pc[env_id].clone()
        if vis_points.shape[-1] == 7:
            vis_points[:, 3:6] = 0
        np.save(f'./plot/temp_{env_id}.npy', vis_points.cpu().numpy())
        return
    
    # captured_img = torch.cat((points_fps[..., :3], points_fps[..., [-1]]), dim=-1)
    # first_frame = torch.where(torch.logical_and(progress_buf==init_timesteps, first_frame[:, 0, 2] == -2).reshape(-1, 1, 1).repeat(1, *first_frame.shape[1:]), captured_img, first_frame)
    gym.end_access_image_tensors(sim)
    return points_fps[..., [0, 1, 2, 6]]


def collect_pointclouds(gym, sim, face_verts, obj_trans, obj_rot, obj_pc, robot_model, robot_state, hand_pos, num_envs, t, camera_props, camera_u2, camera_v2, env_origin, camera_depth_tensor_list, camera_rgb_tensor_list, camera_seg_tensor_list, camera_vinv_mat_list, camera_proj_mat_list, device, num_cameras=1):

    # TODO: i will create a config file later
    x_n_bar = -5
    x_p_bar = 5
    y_n_bar = -5
    y_p_bar = 5
    z_n_bar = 0.6075
    z_p_bar = 5
    depth_bar = 5
    hand_sample_num = 1024
    obj_sample_num = 256
    num_pc_downsample = 1280
    num_pc_presample = 65536
    thres_tactile = 0.005

    robot_translation = torch.zeros([len(robot_state), 3], dtype=torch.float, device=robot_state.device)
    robot_rotation = torch.eye(3, dtype=torch.float, device=robot_state.device).expand(len(robot_state), 3, 3)
    robot_pose = torch.cat([
        robot_translation,
        robot_rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        robot_state[:, :22], 
    ], dim=-1)
    robot_model.set_parameters(robot_pose)
    robot_pc_fps = robot_model.get_surface_points()
    tactile_points = robot_model.get_tactile_points()
    tactile_points_object_frame = (tactile_points - obj_trans[:, None]) @ obj_rot
    in_contact = torch.zeros_like(tactile_points[..., 0])
    # for i in range(num_envs):
    #     dis, dis_signs, _, _ = compute_sdf(tactile_points_object_frame[i], face_verts[i])
    #     dis = torch.sqrt(dis) * dis_signs
    #     in_contact[i] = (dis<thres_tactile)

    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    def split_first_dim(tensor, camera_num):
        return tensor.reshape(-1, camera_num, *tensor.shape[1:])

    depth_tensor = split_first_dim(torch.stack(camera_depth_tensor_list), num_cameras)
    rgb_tensor = split_first_dim(torch.stack(camera_rgb_tensor_list), num_cameras)
    seg_tensor = split_first_dim(torch.stack(camera_seg_tensor_list), num_cameras)
    vinv_mat = split_first_dim(torch.stack(camera_vinv_mat_list), num_cameras)
    proj_matrix = split_first_dim(torch.stack(camera_proj_mat_list), num_cameras)

    def visualize_sensors(env_id=0, t=0):
        COLOR3 =  torch.tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], device=device)
        rgb_img = rgb_tensor[..., :3].to(device)
        background = torch.zeros_like(rgb_img)
        background[..., 0] = 255
        seg_img = COLOR3[seg_tensor.long().to(device)].to(device)
        rgb_img = torch.where((seg_tensor==0).unsqueeze(-1).to(device), background, rgb_img)
        depth_img = depth_tensor.to(device)
        def save_img2(name, rgb_img, seg_img, dep_img):
            import cv2
            import os
            save_dir = 'tmp/vis/diffusion'
            os.makedirs(save_dir, exist_ok=True)
            # cv2.imwrite(osp.join(save_dir, f'{name}_depth.png'), dep_img)
            cv2.imwrite(os.path.join(save_dir, f'{name}_rgb.png'), rgb_img)
            # cv2.imwrite(osp.join(save_dir, f'{name}_seg.png'), seg_img)
            return
        rgb_img = rgb_tensor[env_id, 0, :, :, :3].cpu().numpy()
        seg_img = COLOR3[seg_tensor[env_id,0].long().to(device)].cpu().numpy()
        dep_img = np.zeros_like(rgb_img)
        dep_img[..., 2] = -100 * depth_img[env_id].cpu().numpy()
        save_img2(f'env_{env_id}_{t}_camera', rgb_img, seg_img, dep_img)
        return

    # print(step_counter)
    # print(root_state_tensor[hand_indices, 0:3])
    # pdb.set_trace()
    # for i in range(num_envs):
    visualize_sensors(0, t)

    point_list = []
    valid_list = []
    for i in range(num_cameras):
        # (num_envs, num_pts, 7) (num_envs, num_pts)
        point, valid = depth_image_to_point_cloud_GPU_batch(depth_tensor[:, i], rgb_tensor[:, i], seg_tensor[:, i],
                                                            vinv_mat[:, i], proj_matrix[:, i],
                                                            camera_u2, camera_v2, camera_props.width,
                                                            camera_props.height, depth_bar, device,
                                                            # z_p_bar, z_n_bar
                                                            )
        point_list.append(point)
        valid_list.append(valid)

        # print(f'camera {i}, ', valid.sum(dim=1))

    # (num_envs, 65536 * num_cameras, 7)
    points = torch.cat(point_list, dim=1)
    depth_mask = torch.cat(valid_list, dim=1)
    points[:, :, :3] -= env_origin.view(num_envs, 1, 3)
    # if headless:
    #     points[:, :, :3] -= env_origin.view(num_envs, 1, 3) * 2
    # else:
    #     points[:, :, :3] -= env_origin.view(num_envs, 1, 3)

    x_mask = (points[:, :, 0] > x_n_bar) * (points[:, :, 0] < x_p_bar)
    y_mask = (points[:, :, 1] > y_n_bar) * (points[:, :, 1] < y_p_bar)
    z_mask = (points[:, :, 2] > z_n_bar) * (points[:, :, 2] < z_p_bar)
    # (num_envs, 65536 * 3)
    valid = depth_mask * x_mask * y_mask * z_mask

    # (num_envs,)
    point_nums = valid.sum(dim=1)
    now = 0
    points_list = []
    # (num_valid_pts_total, 7)
    valid_points = points[valid]

    # presample, make num_pts equal for each env
    for env_id, point_num in enumerate(point_nums):
        if point_num == 0:
            print(f'env{env_id}_____point_num = 0_____')
            continue
        points_all = valid_points[now: now + point_num]
        random_ids = torch.randint(0, points_all.shape[0], (num_pc_presample,), device=device,
                                    dtype=torch.long)
        points_all_rnd = points_all[random_ids]
        points_list.append(points_all_rnd)
        now += point_num
    
    assert len(points_list) == num_envs, f'{num_envs - len(points_list)} envs have 0 point'
    # (num_envs, num_pc_presample)
    points_batch = torch.stack(points_list)

    assert hand_sample_num + obj_sample_num == num_pc_downsample

    first_indices = torch.zeros((num_envs, num_pc_presample), device=device, dtype=torch.bool)
    arange = torch.arange(num_envs, device=device)
    min_dist_idx = (points_batch[:,:,:3] - hand_pos[:, None]).norm(dim=-1).argmin(dim=1)
    first_indices[arange, min_dist_idx] = True # in case there are no obj points
    hand_mask = torch.logical_or(first_indices, points_batch[:, :, -1] == 3)
    object_mask = torch.logical_or(first_indices, points_batch[:, :, -1] == 2)
    hand_pc = rand_select(points_batch, hand_mask, num_pc_presample)
    object_pc = rand_select(points_batch, object_mask, num_pc_presample)
    #hand_pc = points_batch.view(-1, 7)[hand_idx]
    #object_idx = torch.where(, idx, zeros)
    #object_pc = points_batch.view(-1, 7)[object_idx]
    hand_fps = sample_points(hand_pc[..., :3], sample_num=hand_sample_num,
                                sample_method='furthest_batch', device=device)
    object_fps = sample_points(object_pc[..., :3], sample_num=obj_sample_num,
                                sample_method='furthest_batch', device=device)
    hand_fps[..., 2] -= TABLE_HEIGHT + ROBOT_BASE_HEIGHT
    object_fps[..., 2] -= TABLE_HEIGHT + ROBOT_BASE_HEIGHT

    obs_fps = torch.cat([hand_fps, object_fps], dim=-2)
    obs_fps = torch.cat([obs_fps, torch.zeros_like(obs_fps[..., [0]])], dim=-1)
    robot_pc_fps = torch.cat([robot_pc_fps, torch.ones_like(robot_pc_fps[..., [0]])], dim=-1)

    points_fps = torch.cat([obs_fps, robot_pc_fps], dim=1)

    def visualize_pc(env_id, camera_id, mode='rgb', pc=points_fps):
        vis_points = pc[env_id].clone()
        if vis_points.shape[-1] == 7:
            vis_points[:, 3:6] = 0
        np.save(f'./plot/temp_{env_id}.npy', vis_points.cpu().numpy())
        return
    
    gym.end_access_image_tensors(sim)
    return points_fps

pointnet2_utils = None
try:
    from pointnet2_ops import pointnet2_utils
except:
    from pytorch3d.ops import sample_farthest_points
    
def sample_points(points, sample_num:int, sample_method:str, device:str):
    if sample_method == 'random':
        raise NotImplementedError

    elif sample_method == "furthest_batch":
        if pointnet2_utils:
            idx = pointnet2_utils.furthest_point_sample(points[:, :, :3].contiguous(), sample_num).long()
        else:
            _, idx = sample_farthest_points(points[:, :, :3].contiguous(), K=sample_num)
            idx = idx.long()
        idx = idx.view(*idx.shape, 1).repeat_interleave(points.shape[-1], dim=2)
        sampled_points = torch.gather(points, dim=1, index=idx)


    elif sample_method == 'furthest':
        eff_points = points[points[:, 2] > 0.04]
        eff_points_xyz = eff_points.contiguous()
        if eff_points.shape[0] < sample_num:
            eff_points = points[:, 0:3].contiguous()
        sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points_xyz.reshape(1, *eff_points_xyz.shape),
                                                                  sample_num)
        sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
    else:
        assert False
    return sampled_points

def rand_select(tensor, mask, target_num):
    row_indices, col_indices = torch.nonzero(mask, as_tuple=True)

    raw_indices = torch.randint(high=2**20, size=(mask.shape[0], target_num), device=mask.device, dtype=torch.long)

    # Calculate the number of occurrences for each unique value
    counts = torch.bincount(row_indices, minlength=mask.shape[0])
    raw_indices = torch.remainder(raw_indices, counts.unsqueeze(-1))
    cumsum_upper = torch.cumsum(counts, dim=0)
    cumsum_lower = cumsum_upper - counts
    indices = cumsum_lower.unsqueeze(-1) + raw_indices

    flat_row, flat_col = row_indices[indices].reshape(-1), col_indices[indices].reshape(-1)
    flat_tensor = tensor[flat_row, flat_col]
    return flat_tensor.reshape(mask.shape[0], target_num, -1)

def depth_image_to_point_cloud_GPU_batch(
        camera_depth_tensor_batch, camera_rgb_tensor_batch,
        camera_seg_tensor_batch, camera_view_matrix_inv_batch,
        camera_proj_matrix_batch, u, v, width: float, height: float,
        depth_bar: float, device: torch.device,
):

    # pdb.set_trace()
    batch_num = camera_depth_tensor_batch.shape[0]

    depth_buffer_batch = mov(camera_depth_tensor_batch, device)
    rgb_buffer_batch = mov(camera_rgb_tensor_batch, device) / 255.0
    seg_buffer_batch = mov(camera_seg_tensor_batch, device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv_batch = camera_view_matrix_inv_batch

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection

    proj_batch = camera_proj_matrix_batch
    fu_batch = 2 / proj_batch[:, 0, 0]
    fv_batch = 2 / proj_batch[:, 1, 1]

    centerU = width / 2
    centerV = height / 2

    Z_batch = depth_buffer_batch

    Z_batch = torch.nan_to_num(Z_batch, posinf=1e10, neginf=-1e10)

    X_batch = -(u.view(1, u.shape[-2], u.shape[-1]) - centerU) / width * Z_batch * fu_batch.view(-1, 1, 1)
    Y_batch = (v.view(1, v.shape[-2], v.shape[-1]) - centerV) / height * Z_batch * fv_batch.view(-1, 1, 1)

    R_batch = rgb_buffer_batch[..., 0].view(batch_num, 1, -1)
    G_batch = rgb_buffer_batch[..., 1].view(batch_num, 1, -1)
    B_batch = rgb_buffer_batch[..., 2].view(batch_num, 1, -1)
    S_batch = seg_buffer_batch.view(batch_num, 1, -1)

    valid_depth_batch = Z_batch.view(batch_num, -1) > -depth_bar

    Z_batch = Z_batch.view(batch_num, 1, -1)
    X_batch = X_batch.view(batch_num, 1, -1)
    Y_batch = Y_batch.view(batch_num, 1, -1)
    O_batch = torch.ones((X_batch.shape), device=device)

    position_batch = torch.cat((X_batch, Y_batch, Z_batch, O_batch, R_batch, G_batch, B_batch, S_batch), dim=1)
    # (b, N, 8)
    position_batch = position_batch.permute(0, 2, 1)
    position_batch[..., 0:4] = position_batch[..., 0:4] @ vinv_batch

    points_batch = position_batch[..., [0, 1, 2, 4, 5, 6, 7]]
    valid_batch = valid_depth_batch  # * valid_z_p_batch * valid_z_n_batch

    return points_batch, valid_batch

def mov(tensor, device):
    return tensor.clone().detach().to(device)

def transform_pcs(pcs):
    if pcs.shape[-1] == 4:
        pcs[..., 3] = pcs[..., 3] - 2
    pcs[..., :3] -= torch.tensor([0., 0., 0.6+ROBOT_BASE_HEIGHT], device=pcs.device) 
    return pcs

def input_wrapper(robot_model, robot_state, rigid_body_state, hand_idx, prev_position, prev_targets, full_obj_pc, obj_trans, obj_rot, img):
    pc_type= 'joint'
    robot_pc_points=1024
    object_pc_points= 256
    with_tactile= 1
    device = robot_state.device
    pc = transform_pcs(img)
    state = torch.cat([obj_trans, obj_rot.reshape(-1, 3*3), robot_state], dim=-1)
    _, tac_pc = get_target_pc(robot_model, state.to(device), full_obj_pc.to(device), robot_pc_points, object_pc_points, pc_type)
    pc = update_seg(pc, tac_pc, with_tac=with_tactile)
    robot_state = torch.cat([robot_state, prev_targets, prev_targets-prev_position], dim=-1)
    return robot_state, pc