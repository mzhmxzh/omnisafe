import torch
from isaacgym import gymapi, gymtorch
import numpy as np
from bisect import bisect
from collections import deque
from utils.isaac_utils import joint_names_isaac, joint_names, get_sim_params, get_plane_params, get_robot_asset_options, get_object_asset_options, create_table_actor, create_robot_actor, init_waiting_pose, get_camera_params, load_cameras, collect_pointclouds, get_point_on_link, points_in_contact, update_sim_params
import copy
# TODO
# 1. when randomizing scale, need to scale the point cloud of objects accordingly.

def get_property_setter_map(gym):
    property_to_setters = {
        "dof_properties": gym.set_actor_dof_properties,
        "tendon_properties": gym.set_actor_tendon_properties,
        "rigid_body_properties": gym.set_actor_rigid_body_properties,
        "rigid_shape_properties": gym.set_actor_rigid_shape_properties,
        "sim_params": gym.set_sim_params,
    }

    return property_to_setters


def get_property_getter_map(gym):
    property_to_getters = {
        "dof_properties": gym.get_actor_dof_properties,
        "tendon_properties": gym.get_actor_tendon_properties,
        "rigid_body_properties": gym.get_actor_rigid_body_properties,
        "rigid_shape_properties": gym.get_actor_rigid_shape_properties,
        "sim_params": gym.get_sim_params,
    }

    return property_to_getters

def get_default_setter_args(gym):
    property_to_setter_args = {
        "dof_properties": [],
        "tendon_properties": [],
        "rigid_body_properties": [True],
        "rigid_shape_properties": [],
        "sim_params": [],
    }

    return property_to_setter_args


def generate_random_samples(attr_randomization_params, shape, curr_gym_step_count,
                            extern_sample=None):

    rand_range = attr_randomization_params['range_sampling']
    distribution = attr_randomization_params['distribution']

    sched_type = attr_randomization_params['schedule'] if 'schedule' in attr_randomization_params else None
    sched_step = attr_randomization_params['schedule_steps'] if 'schedule' in attr_randomization_params else None

    operation = attr_randomization_params['delta_style']

    if sched_type == 'linear':
        sched_scaling = 1 / sched_step * min(curr_gym_step_count, sched_step)
    elif sched_type == 'constant':
        sched_scaling = 0 if curr_gym_step_count < sched_step else 1
    else:
        sched_scaling = 1

    if extern_sample is not None:
        sample = extern_sample

        if operation == 'additive':
            sample *= sched_scaling
        elif operation == 'scaling':
            sample = sample * sched_scaling + 1 * (1 - sched_scaling)

    elif distribution == "gaussian":

        mu, var = (rand_range[0]+rand_range[1])/2, (rand_range[1]-rand_range[0])/2

        if operation == 'additive':
            mu *= sched_scaling
            var *= sched_scaling
        elif operation == 'scaling':
            var = var * sched_scaling  # scale up var over time
            mu = mu * sched_scaling + 1 * (1 - sched_scaling)  # linearly interpolate
        sample = np.random.normal(mu, var, shape)

    elif distribution == "loguniform":

        lo, hi = rand_range
        if operation == 'additive':
            lo *= sched_scaling
            hi *= sched_scaling
        elif operation == 'scaling':
            lo = lo * sched_scaling + 1 * (1 - sched_scaling)
            hi = hi * sched_scaling + 1 * (1 - sched_scaling)

        assert lo > 0, attr_randomization_params
        assert hi > 0, attr_randomization_params
        sample = np.exp(np.random.uniform(np.log(lo), np.log(hi), shape))

    elif distribution == "uniform":

        lo, hi = rand_range
        if operation == 'additive':
            lo *= sched_scaling
            hi *= sched_scaling
        elif operation == 'scaling':
            lo = lo * sched_scaling + 1 * (1 - sched_scaling)
            hi = hi * sched_scaling + 1 * (1 - sched_scaling)
        sample = np.random.uniform(lo, hi, shape)

    return sample

def apply_random_samples(prop, og_prop, attr, attr_randomization_params,
                         curr_gym_step_count, extern_sample=None, bucketing_randomization_params=None):
    
    """
    @params:
        prop: property we want to randomise
        og_prop: the original property and its value 
        attr: which particular attribute we want to randomise e.g. damping, stiffness
        attr_randomization_params: the attribute randomisation meta-data e.g. distr, range, schedule
        curr_gym_step_count: gym steps so far 

    """

    if isinstance(prop, gymapi.SimParams):
        
        if attr == 'gravity':
            sample = generate_random_samples(attr_randomization_params, 3, curr_gym_step_count)
            if attr_randomization_params['delta_style'] == 'scaling':
                prop.gravity.x = og_prop['gravity'].x * sample[0]
                prop.gravity.y = og_prop['gravity'].y * sample[1]
                prop.gravity.z = og_prop['gravity'].z * sample[2]
            elif attr_randomization_params['delta_style'] == 'additive':
                #prop.gravity.x = og_prop['gravity'].x + sample[0]
                #prop.gravity.y = og_prop['gravity'].y + sample[1]
                prop.gravity.z = og_prop['gravity'].z + sample[2]


        if attr == 'rest_offset':

           sample = generate_random_samples(attr_randomization_params, 1, curr_gym_step_count)
           prop.physx.rest_offset = sample
                

    elif isinstance(prop, np.ndarray):
        sample = generate_random_samples(attr_randomization_params, prop[attr].shape,
                                         curr_gym_step_count, extern_sample)

        if attr_randomization_params['delta_style'] == 'scaling':
            new_prop_val = og_prop[attr] * sample
        elif attr_randomization_params['delta_style'] == 'additive':
            new_prop_val = og_prop[attr] + sample

        if 'num_buckets' in attr_randomization_params and attr_randomization_params['num_buckets'] > 0:
            new_prop_val = get_bucketed_val(new_prop_val, attr_randomization_params)
        prop[attr] = new_prop_val
    else:
        sample = generate_random_samples(attr_randomization_params, 1,
                                         curr_gym_step_count, extern_sample)
        cur_attr_val = og_prop[attr]
        if attr_randomization_params['delta_style'] == 'scaling':
            new_prop_val = cur_attr_val * sample
        elif attr_randomization_params['delta_style'] == 'additive':
            new_prop_val = cur_attr_val + sample

        if 'num_buckets' in attr_randomization_params and attr_randomization_params['num_buckets'] > 0:
            if bucketing_randomization_params is None:
                new_prop_val = get_bucketed_val(new_prop_val, attr_randomization_params)
            else:
                new_prop_val = get_bucketed_val(new_prop_val, bucketing_randomization_params)
        setattr(prop, attr, new_prop_val)

def update_delay_queue(delay_queue, current_state, adr_params):
    if len(delay_queue) == 0:
        delay_queue.append(current_state)
        return delay_queue, current_state
    delay_max = adr_params['delay_max']
    delay_prob = adr_params['delay_prob']
    delay_steps = (np.random.rand(delay_max) < delay_prob).sum()
    
    if len(delay_queue) < delay_steps + 1:
        current_state = delay_queue[-1]
    else:
        current_state = delay_queue[delay_steps]
        
    if len(delay_queue) < delay_max:
        delay_queue = [current_state] + delay_queue
    else:
        delay_queue = [current_state] + delay_queue[:-1]
        
    return delay_queue, current_state

def init_adr_params(env, config, adr_cfg):
        env.need_rerandomize = torch.full((config.num_envs,), 1, device=env.device).bool()
        env.worker_adr_boundary_fraction = adr_cfg["worker_adr_boundary_fraction"]
        env.adr_queue_threshold_length = adr_cfg["adr_queue_threshold_length"]
        env.adr_objective_threshold_low = adr_cfg["adr_objective_threshold_low"]
        env.adr_objective_threshold_high = adr_cfg["adr_objective_threshold_high"]
        env.adr_extended_boundary_sample = adr_cfg["adr_extended_boundary_sample"]
        env.adr_rollout_perf_alpha = adr_cfg["adr_rollout_perf_alpha"]
        env.update_adr_ranges = adr_cfg["update_adr_ranges"]
        env.adr_clear_other_queues = adr_cfg["clear_other_queues"]
        env.adr_rollout_perf_last = None
        env.adr_load_from_checkpoint = adr_cfg["adr_load_from_checkpoint"]
        env.adr_first_randomize = True
        env.original_props = {}
        env.npd = 0.0
        
        # 0 = rollout worker 1 = ADR worker 2 = eval worker (see https://arxiv.org/pdf/1910.07113.pdf Section 5)
        env.worker_types = torch.zeros(config.num_envs, dtype=torch.long, device=env.device)
        env.adr_gravity = adr_cfg["adr_gravity"]
        env.adr_gravity["range"] = env.adr_gravity["init_range"]
        env.adr_params = adr_cfg["params"]
        env.adr_params_keys = []
        env.num_adr_params = 0
        for actor, actor_value in env.adr_params.items():
            for k, k_value in actor_value.items():
                if k == 'scale':
                    env.adr_params[actor][k]["range"] = copy.deepcopy(env.adr_params[actor][k]["init_range"])
                    env.adr_params[actor][k]["range_sampling"] = copy.deepcopy(env.adr_params[actor][k]["init_range"])
                    if "limits" not in env.adr_params[actor][k]:
                        env.adr_params[actor][k]["limits"] = [None, None]
                    if "delta_style" in env.adr_params[actor][k]:
                        assert env.adr_params[actor][k]["delta_style"] in ["additive", "scaling"]
                    else:
                        env.adr_params[actor][k]["delta_style"] = "additive"
                    param_type = env.adr_params[actor][k].get("type", "uniform")
                    dtype = torch.long if param_type == "categorical" else torch.float
                    env.num_adr_params += 1
                    env.adr_params_keys.append(f"{actor}-{k}")
                else:
                    for attr, _ in k_value.items():
                        env.adr_params[actor][k][attr]["range"] = copy.deepcopy(env.adr_params[actor][k][attr]["init_range"])
                        env.adr_params[actor][k][attr]["range_sampling"] = copy.deepcopy(env.adr_params[actor][k][attr]["init_range"])
                        if "limits" not in env.adr_params[actor][k][attr]:
                            env.adr_params[actor][k][attr]["limits"] = [None, None]
                        if "delta_style" in env.adr_params[actor][k][attr]:
                            assert env.adr_params[actor][k][attr]["delta_style"] in ["additive", "scaling"]
                        else:
                            env.adr_params[actor][k][attr]["delta_style"] = "additive"
                        param_type = env.adr_params[actor][k][attr].get("type", "uniform")
                        dtype = torch.long if param_type == "categorical" else torch.float
                        env.num_adr_params += 1
                        env.adr_params_keys.append(f"{actor}-{k}-{attr}")
        # modes for ADR workers. 
        # there are 2n valid mode values, where mode 2n is lower range and mode 2n+1 is upper range for DR parameter n
        env.adr_modes = torch.zeros(config.num_envs, dtype=torch.long, device=env.device)
        env.adr_objective_queues = [deque(maxlen=env.adr_queue_threshold_length) for _ in range(2*env.num_adr_params)]
        
        # related to action delay
        env.adr_delay_max = adr_cfg["delay_max"]
        env.adr_delay_prob = adr_cfg["delay_prob"]
        env.adr_last_action_queue = []

def update_adr_param_and_rerandomize(env, indices, RolloutWorkerModes):
    total_nats = 0.0
    boundary_sr = 0.0
    env.adr_ranges = {}
    if not env.adr_first_randomize:
        # TODO: check buffers and update bounds
        success_vec = env.record_success
        rand_env_mask = torch.zeros(env.config.num_envs, dtype=torch.bool, device=env.device)
        rand_env_mask[indices] = True
        adr_params_iter = list(enumerate(env.adr_params_keys))
        
        # check ADR buffer parameter-wise
        for n, adr_param_name in adr_params_iter:
            if adr_param_name == 'gravity':
                assert (env.adr_gravity['range'][1] - env.adr_gravity['range'][0]) >= 0, 'gravity'
                env.adr_ranges['gravity'] = env.adr_gravity['range']
                total_nats += (env.adr_gravity['range'][1] - env.adr_gravity['range'][0])
                pass
            low_idx = 2*n
            high_idx = 2*n+1
            adr_workers_low = torch.logical_and(env.worker_types == RolloutWorkerModes.ADR_BOUNDARY, env.adr_modes == low_idx)
            adr_workers_high = torch.logical_and(env.worker_types == RolloutWorkerModes.ADR_BOUNDARY, env.adr_modes == high_idx)
            # environments which will be evaluated for ADR
            adr_done_low = torch.logical_and(rand_env_mask, adr_workers_low) 
            adr_done_high = torch.logical_and(rand_env_mask, adr_workers_high)
            
            objective_low_bounds = success_vec[adr_done_low].float()
            objective_high_bounds = success_vec[adr_done_high].float()
            # below tries relaxation [which may be WRONG] of boundary condition
            # objective_low_bounds = success_vec[adr_workers_low].float()
            # objective_high_bounds = success_vec[adr_workers_high].float()
            # add the success of objectives to queues
        
            env.adr_objective_queues[low_idx].extend(objective_low_bounds.cpu().numpy().tolist())
            env.adr_objective_queues[high_idx].extend(objective_high_bounds.cpu().numpy().tolist())
            low_queue = env.adr_objective_queues[low_idx]
            high_queue = env.adr_objective_queues[high_idx]

            mean_low = np.mean(low_queue) if len(low_queue) > 0 else 0.
            mean_high = np.mean(high_queue) if len(high_queue) > 0 else 0.
            boundary_sr += mean_low
            boundary_sr += mean_high
            # if len(low_queue) > 0:
            #     print(adr_param_name.split('-')[-1], "low",  low_queue)
            # if len(high_queue) > 0:
            #     print(adr_param_name.split('-')[-1], "high", high_queue)
            #print(f"attr={adr_param_name.split('-')[-1]}, low={mean_low}, high={mean_high}")

            split_param_name = adr_param_name.split('-')
            if split_param_name[-1] == "scale":
                actor, k = split_param_name[0], split_param_name[1]
                current_range = env.adr_params[actor][k]["range"]
                range_lower = current_range[0]
                range_upper = current_range[1]

                range_limits = env.adr_params[actor][k]["limits"]
                init_range = env.adr_params[actor][k]["init_range"]
                default = env.adr_params[actor][k]["default"]
                adr_param_dict = env.adr_params[actor][k]
            else:
                actor, k, attr = split_param_name[0], split_param_name[1], split_param_name[2]
                current_range = env.adr_params[actor][k][attr]["range"]
                range_lower = current_range[0]
                range_upper = current_range[1]

                range_limits = env.adr_params[actor][k][attr]["limits"]
                init_range = env.adr_params[actor][k][attr]["init_range"]
                default = env.adr_params[actor][k][attr]["default"]
                adr_param_dict = env.adr_params[actor][k][attr]                    
            changed_low = False
            changed_high = False
            objective_low = env.adr_objective_threshold_low
            objective_high = env.adr_objective_threshold_high
            if len(low_queue) >= env.adr_queue_threshold_length:
                print(f"mean_low:{mean_low}")
                if mean_low < objective_low:
                    range_lower, changed_low = modify_adr_param(
                        env, range_lower, 'up', adr_param_dict, param_limit=default[0]
                    )

                elif mean_low > objective_high:
                    range_lower, changed_low = modify_adr_param(
                        env, range_lower, 'down', adr_param_dict, param_limit=range_limits[0]
                    )
                    
                env.adr_objective_queues[low_idx].clear()
                if changed_low:
                    env.worker_types[adr_workers_low] = RolloutWorkerModes.ADR_ROLLOUT
                        
            if len(high_queue) >= env.adr_queue_threshold_length:
                print(f"mean_high:{mean_high}")
                if mean_high < objective_low:
                    range_upper, changed_high = modify_adr_param(
                        env, range_upper, 'down', adr_param_dict, param_limit=default[1]
                    )
                elif mean_high > objective_high:
                    range_upper, changed_high = modify_adr_param(
                        env, range_upper, 'up', adr_param_dict, param_limit=range_limits[1]
                    )
            
                env.adr_objective_queues[high_idx].clear()
                if changed_high:
                    env.worker_types[adr_workers_high] = RolloutWorkerModes.ADR_ROLLOUT

            if split_param_name[-1] == "scale":
                actor, k = split_param_name[0], split_param_name[1]
                env.adr_params[actor][k]["range"] = [range_lower, range_upper]
                env.adr_params[actor][k]["range_sampling"] = [range_lower, range_upper]
                total_nats += (env.adr_params[actor][k]["range"][1] - env.adr_params[actor][k]["range"][0])
                env.adr_ranges[adr_param_name] = env.adr_params[actor][k]["range"]
            else:
                actor, k, attr = split_param_name[0], split_param_name[1], split_param_name[2]
                env.adr_params[actor][k][attr]["range"] = [range_lower, range_upper]
                env.adr_params[actor][k][attr]["range_sampling"] = [range_lower, range_upper]
                total_nats += (env.adr_params[actor][k][attr]["range"][1] - env.adr_params[actor][k][attr]["range"][0])
                env.adr_ranges[adr_param_name] = env.adr_params[actor][k][attr]["range"]

    # calculate NPD
    env.npd = total_nats / (env.num_adr_params + 1)
    env.boundary_sr = boundary_sr / (env.num_adr_params * 2)
    # allocate worker modes
    new_worker_types = torch.zeros_like(indices, device=env.device, dtype=torch.long)
    worker_types_rand = torch.rand_like(indices, device=env.device, dtype=torch.float)
    new_worker_types[worker_types_rand < env.worker_adr_boundary_fraction] = RolloutWorkerModes.ADR_BOUNDARY
    new_worker_types[worker_types_rand >= env.worker_adr_boundary_fraction] = RolloutWorkerModes.ADR_ROLLOUT
    # newly assigned boundary workers need to re-randomize
    env.need_rerandomize[indices[worker_types_rand < env.worker_adr_boundary_fraction]] = True
    env.worker_types[indices] = new_worker_types
    # resample the ADR modes (which boundary values to sample) for the given environments (only applies to ADR_BOUNDARY mode)
    env.adr_modes[indices] = torch.randint(0, env.num_adr_params * 2, (indices.shape[0],), dtype=torch.long, device=env.device)
    for ind in indices:
        env_type = env.worker_types[ind]
        if env_type == RolloutWorkerModes.TEST_ENV:  # eval worker, uses default fixed params
            raise NotImplementedError
        else:
            if env_type == RolloutWorkerModes.ADR_BOUNDARY: # ADR worker, substitute upper or lower bound as entire range for this env
                adr_mode = env.adr_modes[ind]
                adr_id = adr_mode // 2 # which adr parameter
                adr_bound = adr_mode % 2 # 0 = lower, 1 = upper
                param_name = env.adr_params_keys[adr_id]
                param_name = param_name.split("-")
                if param_name[-1] == "scale":
                    actor, k = param_name[0], param_name[1]
                    # TODO: update corresponding param range to bound
                    #boundary_value = env.adr_params[actor][k]["range"][adr_bound]
                    boundary_value = 0.9 * env.adr_params[actor][k]["range"][adr_bound] + 0.1 * env.adr_params[actor][k]["range"][1-adr_bound]
                    env.adr_params[actor][k]['range_sampling'][0] = boundary_value
                    env.adr_params[actor][k]['range_sampling'][1] = boundary_value
                else:
                    actor, k, attr = param_name[0], param_name[1], param_name[2]
                    # TODO: update corresponding param range to bound
                    #boundary_value = env.adr_params[actor][k][attr]["range"][adr_bound]
                    boundary_value = 0.9 * env.adr_params[actor][k][attr]["range"][adr_bound] + 0.1 * env.adr_params[actor][k][attr]["range"][1-adr_bound]
                    env.adr_params[actor][k][attr]['range_sampling'][0] = boundary_value
                    env.adr_params[actor][k][attr]['range_sampling'][1] = boundary_value
                #raise ValueError(env.adr_params)
            adr_params = env.adr_params
        # only at certain probability do we re-randomize physics param when the worker is for training use.
        # we hypothesis doing so may be helpful for stable training
        if (np.random.rand(1)[0] < env.config.reset_randomize_physics_prob):
        # if not env.need_rerandomize[ind]:
        #     continue
            if env.worker_types[ind] == RolloutWorkerModes.ADR_ROLLOUT:
                apply_randomization(env, ind, adr_params, boundary=True)
            else:
                apply_randomization(env, ind,adr_params)
        if env_type == RolloutWorkerModes.ADR_BOUNDARY:
            if param_name[-1] == "scale":
                env.adr_params[actor][k]['range_sampling'] = copy.deepcopy(env.adr_params[actor][k]['range'])
            else:
                env.adr_params[actor][k][attr]['range_sampling'] = copy.deepcopy(env.adr_params[actor][k][attr]['range'])
        env.need_rerandomize[ind] = False
        env.refresh()
    
    # sample param values
    env.adr_first_randomize = False

def update_manualdr_param_and_rerandomize(sim_env, indices):
    for env_id in sim_env.env_ids[indices]:
        # if not sim_env.need_rerandomize[env_id]:
        #     continue
        env = sim_env.envs[env_id]
        for j in range(sim_env.config.num_obj_per_env):
            # update object properties. See isaacgym.gymapi.RigidShapeProperties
            object_handle = sim_env.obj_handles[env_id][j]
            # scale
            new_scale = sim_env.config.scale_range[0] + np.random.rand(1)[0]* (sim_env.config.scale_range[1] - sim_env.config.scale_range[0])
            sim_env.gym.set_actor_scale(env, object_handle, new_scale)
            # friction, restitution
            object_shape_props = sim_env.gym.get_actor_rigid_shape_properties(env, object_handle)
            object_body_props = sim_env.gym.get_actor_rigid_body_properties(env, object_handle)
            object_shape_props[0].friction = sim_env.config.friction * (sim_env.config.friction_range[0] + np.random.rand(1)[0]* (sim_env.config.friction_range[1] - sim_env.config.friction_range[0]))
            # yet to check if default value is 1.0
            object_shape_props[0].restitution = sim_env.obj_original_restitution * (sim_env.config.restitution_range[0] + np.random.rand(1)[0]* (sim_env.config.restitution_range[1] - sim_env.config.restitution_range[0]))
            # mass (multiplicative)
            object_body_props[0].mass = sim_env.obj_original_masses[env_id][j] * (np.random.rand(1)[0]* (sim_env.config.mass_range[1] - sim_env.config.mass_range[0]) + sim_env.config.mass_range[0])
            sim_env.gym.set_actor_rigid_shape_properties(env, object_handle, object_shape_props)
            sim_env.gym.set_actor_rigid_body_properties(env, object_handle, object_body_props)

        # update robot properties.
        robot_handle = sim_env.robot_handles[env_id]
        robot_dof_props = sim_env.gym.get_actor_dof_properties(env, robot_handle)
        robot_shape_props = sim_env.gym.get_actor_rigid_shape_properties(env, robot_handle)
        
        robot_shape_props[0].friction = sim_env.config.friction * (sim_env.config.friction_range[0] + np.random.rand(1)[0]* (sim_env.config.friction_range[1] - sim_env.config.friction_range[0]))
        # yet to check if default value is 1.0
        robot_shape_props[0].restitution = sim_env.obj_original_restitution * (sim_env.config.restitution_range[0] + np.random.rand(1)[0]* (sim_env.config.restitution_range[1] - sim_env.config.restitution_range[0]))
        robot_dof_props['stiffness'] = sim_env.original_dof_prop['stiffness'] * np.exp(np.random.uniform(np.log(sim_env.config.stiffness_range[0]), np.log(sim_env.config.stiffness_range[1]), (1,)))[0]
        robot_dof_props['damping'] = sim_env.original_dof_prop['damping'] * np.exp(np.random.uniform(np.log(sim_env.config.damping_range[0]), np.log(sim_env.config.damping_range[1]), (1,)))[0]
        robot_dof_props['effort'] = sim_env.original_dof_prop['effort'] * (sim_env.config.effort_range[0] + np.random.rand(1)[0]* (sim_env.config.effort_range[1] - sim_env.config.effort_range[0]))

        sim_env.gym.set_actor_rigid_shape_properties(env, robot_handle, robot_shape_props)
        sim_env.gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)
        sim_env.need_rerandomize[env_id] = False

def rerandomize_physics_gravity(env, step_cnt):
    if (step_cnt % env.config.randomize_gravity_every) == 0:
        if env.use_adr:
            sim_params = get_sim_params(env.config)
            success_mean = env.record_success.float().mean()
            delta = env.adr_gravity['delta']
            range_low, range_high = env.adr_gravity['range']
            limit_low, limit_high = env.adr_gravity['limits']
            if success_mean < env.adr_objective_threshold_low:
                range_low = min(range_low + delta, 0.0)
                range_high = max(range_high - delta, 0.0)
            elif success_mean >= env.adr_objective_threshold_high:
                range_low = max(range_low - delta, limit_low)
                range_high = min(range_high + delta, limit_high)
            env.adr_gravity['range'][0] = range_low
            env.adr_gravity['range'][1] = range_high
            mu, var = (range_low+range_high)/2.0, (range_high-range_low)/2.0
            sample = np.random.normal(mu, var, (1,))[0]
            sim_params.gravity.z = sim_params.gravity.z + sample
            env.gym.set_sim_params(env.sim, sim_params)
        else:
            # randomize sim params
            sim_params = get_sim_params(env.config)
            sim_params = update_sim_params(sim_params, env.config)
            env.gym.set_sim_params(env.sim, sim_params)

def apply_randomization(sim_env, ind, adr_params, boundary=False):
    # sample and set all parameters that are randomized
    env = sim_env.envs[ind]
    param_setters_map = get_property_setter_map(sim_env.gym)
    param_setter_defaults_map = get_default_setter_args(sim_env.gym)
    param_getters_map = get_property_getter_map(sim_env.gym)
    for actor, actor_properties in adr_params.items():
        # actor = ['robot','object']
        # actor = dict of specific params to randomize
        handle = sim_env.gym.find_actor_handle(env, actor)
        for prop_name, prop_attrs in actor_properties.items():
            # prop_name = 'friction', 'mass', etc
            # prop_attrs = dict of attribute and value pairs
            if prop_name == 'scale':
                raise NotImplementedError()
                # TODO: when randomizing scale, need to scale the point cloud of objects accordingly.
                attr_randomization_params = prop_attrs
                sample = generate_random_samples(attr_randomization_params, 1,
                                                    sim_env.curr_iter, None)
                og_scale = 1.0
                if attr_randomization_params['delta_style'] == 'scaling':
                    new_scale = og_scale * sample
                elif attr_randomization_params['delta_style'] == 'additive':
                    new_scale = og_scale + sample
                if not sim_env.gym.set_actor_scale(env, handle, new_scale):
                    raise ValueError(f"set scale failed: actor={actor}, actor_properties={actor_properties}")
                continue
            # if list it is likely to be 
            #  - rigid_body_properties
            #  - rigid_shape_properties
            prop = param_getters_map[prop_name](env, handle)
            if isinstance(prop, list):
                if sim_env.adr_first_randomize:
                    sim_env.original_props[prop_name] = [
                        {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                for attr, attr_randomization_params in prop_attrs.items():
                    # here relaxes the boundary resampling: we only resample that specific param and fix others, in order to minimize shift of physical properties
                    # if boundary and (not (attr_randomization_params['range_sampling'][1] == attr_randomization_params['range_sampling'][0])):
                    #     continue
                    for body_idx, (p, og_p) in enumerate(zip(prop, sim_env.original_props[prop_name])):
                        curr_prop = p 
                        apply_random_samples(
                            curr_prop, og_p, attr, attr_randomization_params,
                            sim_env.curr_iter, None,)
            # if it is not a list, it is likely an array 
            # which means it is for dof_properties
            else:
                if sim_env.adr_first_randomize:
                    sim_env.original_props[prop_name] = deepcopy(prop)
                for attr, attr_randomization_params in prop_attrs.items():
                    # here relaxes the boundary resampling: we only resample that specific param and fix others, in order to minimize shift of physical properties
                    # if boundary and (not (attr_randomization_params['range_sampling'][1] == attr_randomization_params['range_sampling'][0])):
                    #     continue
                    apply_random_samples(
                            prop, sim_env.original_props[prop_name], attr,
                            attr_randomization_params, sim_env.curr_iter, None)
            setter = param_setters_map[prop_name]
            default_args = param_setter_defaults_map[prop_name]
            setter(env, handle, prop)

def modify_adr_param(env, param, direction, adr_param_dict, param_limit=None):
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
        assert not new_val == param, adr_param_dict
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
        assert not new_val == param, adr_param_dict
        if param_limit is not None:
            new_val = max(new_val, param_limit)
        
        changed = abs(new_val - param) > 1e-9
        
        return new_val, changed
    else:
        raise NotImplementedError