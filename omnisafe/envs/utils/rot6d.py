import torch

epison = 1e-8


def normalize(q):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1, keepdim=True)
    return q.div(norm+epison)


def assert_normalized(q, atol=1e-3):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1)
    norm_check =  (norm - 1.0).abs()
    try:
        assert torch.max(norm_check) < atol
    except:
        print("normalization failure: {}.".format(torch.max(norm_check)))
        return -1
    return 0


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out


def unit_quaternion_to_matrix(q):
    assert_normalized(q)
    w, x, y, z= torch.unbind(q, dim=-1)
    matrix = torch.stack(( 1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y* w,
                        2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w,  
                        2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x -2*y*y),
                        dim=-1)
    matrix_shape = list(matrix.shape)[:-1]+[3,3]
    return matrix.view(matrix_shape).contiguous()


def axis_theta_to_quater(axis, theta):  # axis: [Bs, 3], theta: [Bs]
    w = torch.cos(theta / 2.)  # [Bs]
    u = torch.sin(theta / 2.)  # [Bs]
    xyz = axis * u.unsqueeze(-1)  # [Bs, 3]
    new_q = torch.cat([w.unsqueeze(-1), xyz], dim=-1)  # [Bs, 4]
    new_q = normalize(new_q)
    return new_q


def axis_theta_to_matrix(axis, theta):
    quater = axis_theta_to_quater(axis, theta)  # [Bs, 4]
    return unit_quaternion_to_matrix(quater)