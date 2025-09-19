import torch
import kornia.geometry as kg


def rotation_matrix_to_quaternion(R: torch.Tensor):
    return kg.conversions.rotation_matrix_to_quaternion(R)


def quaternion_to_rotation_matrix(q: torch.Tensor):
    return kg.conversions.quaternion_to_rotation_matrix(q)


def zero_z_coordinate(xyz: torch.Tensor):
    mask = torch.tensor([1, 1, 0], dtype=xyz.dtype, device=xyz.device).view(1, 3)
    xyz = xyz * mask
    return xyz


def zero_z_tilt(q: torch.Tensor):
    euler = kg.conversions.quaternion_to_axis_angle(q)
    euler = zero_z_coordinate(euler)
    q_new = kg.conversions.axis_angle_to_quaternion(euler)
    return q_new


def zero_z_scale(S: torch.Tensor, R: torch.Tensor):
    # R is from gaussian to world to plane
    S_diag = torch.diag_embed(S)
    S_diag = torch.bmm(R, torch.bmm(S_diag, R.transpose(1, 2)))
    S_diag[:, :, 2].clamp_(min=1e-6, max=1e-3)
    S_diag[:, 2, :].clamp_(min=1e-6, max=1e-3)
    S_diag = torch.bmm(R.transpose(1, 2), torch.bmm(S_diag, R))
    S = torch.diagonal(S_diag, dim1=-2, dim2=-1)
    S.clamp_(min=1e-6)
    return S


def xyz_world_to_planes(xyz: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    # R is from plane to world
    # R_inverse is from world to plane
    origin = xyz - t
    R_inverse = R.transpose(0, 1)
    xyz = torch.matmul(R_inverse, origin.unsqueeze(-1)).squeeze(-1)
    return xyz


def xyz_world_to_planes_batch(
    xyz: torch.Tensor, R_batch: torch.Tensor, t_batch: torch.Tensor
):
    origin = xyz - t_batch
    R_inverse = R_batch.transpose(1, 2)
    xyz = torch.bmm(R_inverse, origin.unsqueeze(-1)).squeeze(-1)
    return xyz


def xyz_planes_to_world(xyz: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    # R is from plane to world
    xyz = torch.matmul(R, xyz.unsqueeze(-1)).squeeze(-1) + t
    return xyz


def xyz_planes_to_world_batch(
    xyz: torch.Tensor, R_batch: torch.Tensor, t_batch: torch.Tensor
):
    xyz = torch.bmm(R_batch, xyz.unsqueeze(-1)).squeeze(-1) + t_batch
    return xyz


def rotation_world_to_planes_batch(
    rotations: torch.Tensor, plane_rotations: torch.Tensor
):
    q1 = plane_rotations[:, 0]
    v1 = -plane_rotations[:, 1:]

    q2 = rotations[:, 0]
    v2 = rotations[:, 1:]

    q = q1 * q2 - torch.sum(v1 * v2, dim=1)
    v = q1.view(-1, 1) * v2 + q2.view(-1, 1) * v1 + torch.cross(v1, v2, dim=1)

    rotation = torch.cat((q.view(-1, 1), v), dim=1)
    return rotation


def rotation_planes_to_world_batch(
    rotations: torch.Tensor, plane_rotations: torch.Tensor
):
    q1 = plane_rotations[:, 0]
    v1 = plane_rotations[:, 1:]

    q2 = rotations[:, 0]
    v2 = rotations[:, 1:]

    q = q1 * q2 - torch.sum(v1 * v2, dim=1)
    v = q1.view(-1, 1) * v2 + q2.view(-1, 1) * v1 + torch.cross(v1, v2, dim=1)

    composed_rotations = torch.cat((q.view(-1, 1), v), dim=1)
    return composed_rotations


def probability_erf(x: torch.Tensor, sigma: float):
    denominator = sigma * torch.sqrt(torch.tensor(2, device="cuda"))
    probabilities = torch.erf(x / denominator)
    return probabilities


def normal_to_quaternion(normal: torch.Tensor):
    normal = normal / torch.norm(normal)

    reference_vector = torch.tensor([0.0, 0.0, 1.0], device="cuda")
    axis = torch.linalg.cross(reference_vector, normal)

    dot_product = torch.dot(reference_vector, normal)
    angle = torch.acos(dot_product)

    if torch.isclose(dot_product, torch.tensor(1.0)):
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")

    if torch.isclose(dot_product, torch.tensor(-1.0)):
        return torch.tensor([0.0, 1.0, 0.0, 0.0], device="cuda")

    axis = axis / torch.norm(axis)

    quat_w = torch.cos(angle / 2)
    quat_xyz = axis * torch.sin(angle / 2)

    quaternion = torch.cat((quat_w.unsqueeze(0), quat_xyz))
    return quaternion


def quaternion_to_normal(q: torch.Tensor):
    q = q / torch.norm(q)
    w, x, y, z = q

    n_x = 2 * (x * z + w * y)
    n_y = 2 * (y * z - w * x)
    n_z = 1 - 2 * (x**2 + y**2)

    normal = torch.tensor([n_x, n_y, n_z], device="cuda")
    normal = normal / torch.norm(normal)
    return normal
