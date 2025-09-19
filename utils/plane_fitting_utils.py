import torch
from arguments import OptimizationParams
from scene.cameras import Camera
from torch_kdtree import build_kd_tree
from utils.plane_utils import normal_to_quaternion


def fit_plane_3_points_batch(points: torch.Tensor):
    p1, p2, p3 = points[:, 0], points[:, 1], points[:, 2]  # (B, 3)

    v1 = p2 - p1
    v2 = p3 - p1

    normal = torch.cross(v1, v2, dim=1)
    normal = normal / torch.norm(normal, dim=1, keepdim=True)

    A, B, C = normal[:, 0], normal[:, 1], normal[:, 2]
    D = -torch.sum(normal * p1, dim=1)
    return torch.stack([A, B, C, D], dim=1)  # (B, 4)


def points_to_plane_distance_batch(points: torch.Tensor, planes: torch.Tensor):
    A, B, C, D = planes[:, 0], planes[:, 1], planes[:, 2], planes[:, 3]
    x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]

    numerator = torch.abs(A[:, None] * x + B[:, None] * y + C[:, None] * z + D[:, None])
    denominator = torch.sqrt(A**2 + B**2 + C**2)[:, None] + 1e-6
    return numerator / denominator  # (B, N)


def ransac(points: torch.Tensor, threshold: float, max_iter: int = 1000):
    N = points.size(0)

    # Sample points in batch
    sampled_indices = torch.randint(0, N, (max_iter, 3), device=points.device)
    sampled_points = points[sampled_indices]

    # Fit planes in batch
    planes = fit_plane_3_points_batch(sampled_points)

    distances = points_to_plane_distance_batch(
        points.unsqueeze(0).expand(max_iter, -1, -1), planes
    )
    inliers = distances < threshold
    inlier_counts = inliers.sum(dim=1)

    best_idx = torch.argmax(inlier_counts)

    return planes[best_idx], inliers[best_idx], distances[best_idx]


def get_pixel_grid(height: int, width: int):
    u = torch.arange(0, width, device="cuda").repeat(height, 1)
    v = torch.arange(0, height, device="cuda").repeat(width, 1).t()

    # Stack to get (height, width, 2)
    pixels = torch.stack((u, v), dim=-1).to(torch.float32)
    pixels += 0.5  # Center of the pixel as in the camera model
    return pixels


def get_xyz_depth_view_h(depth: torch.Tensor, cam: Camera):
    pixels = get_pixel_grid(depth.shape[0], depth.shape[1])

    pixels = pixels.reshape(-1, 2)
    depth = depth.reshape(-1)

    # Convert to camera coordinates
    xyz_view_h = torch.stack(
        [
            (pixels[:, 0] - cam.Cx) * depth / cam.Fx,
            (pixels[:, 1] - cam.Cy) * depth / cam.Fy,
            depth,
            torch.ones_like(depth),
        ],
        dim=1,
    )

    return xyz_view_h


def get_xyz_depth(depth: torch.Tensor, cam: Camera):
    xyz_view_h = get_xyz_depth_view_h(depth, cam)

    # Convert to world coordinates
    view_world_transform = cam.world_view_transform.inverse()
    xyz_world = xyz_view_h @ view_world_transform
    xyz_world = xyz_world[:, :3] / xyz_world[:, 3].unsqueeze(1)

    return xyz_world


def get_xyz_view_h(xyz_world: torch.Tensor, cam: Camera):
    xyz_world_h = torch.cat(
        [
            xyz_world,
            torch.ones(xyz_world.shape[0], 1, device="cuda"),
        ],
        dim=1,
    )
    xyz_view_h = xyz_world_h @ cam.world_view_transform
    return xyz_view_h


def get_image_pixels(xyz_world: torch.Tensor, cam: Camera):

    xyz_view_h = get_xyz_view_h(xyz_world, cam)
    xyz_view = xyz_view_h[:, :3]

    # Convert xyz to uv image pixel coordinates
    xyz_view = xyz_view / xyz_view[:, 2].unsqueeze(1)

    K = torch.tensor(
        [
            [cam.Fx, 0, cam.Cx],
            [0, cam.Fy, cam.Cy],
            [0, 0, 1],
        ],
        device="cuda",
    )

    image_pixels = xyz_view @ K.transpose(0, 1)
    image_pixels = image_pixels[:, :2]

    return image_pixels


def build_image_pixels_kdtree(xyz_world: torch.Tensor, cam: Camera):
    image_pixels = get_image_pixels(xyz_world, cam)
    kdtree = build_kd_tree(image_pixels)
    return kdtree


@torch.no_grad()
def find_closest_mask_points(
    mask: torch.Tensor, reloc_pixels: torch.Tensor, reloc_filter: torch.Tensor
):

    mask = mask > 0.5
    H, W = mask.shape

    reloc_pixels = torch.round(reloc_pixels).to(torch.int32)
    reloc_indices = torch.where(reloc_filter)[0]

    valid_mask = (
        (reloc_pixels[:, 0] >= 0)
        & (reloc_pixels[:, 0] < W)
        & (reloc_pixels[:, 1] >= 0)
        & (reloc_pixels[:, 1] < H)
    )

    valid_reloc_pixels = reloc_pixels[valid_mask]
    valid_indices = torch.where(valid_mask)[0]

    # Check which points are inside the mask
    inside_mask = mask[valid_reloc_pixels[:, 1], valid_reloc_pixels[:, 0]]
    inside_indices = valid_indices[inside_mask]

    mask_xyz = torch.zeros_like(reloc_filter, device="cuda")
    mask_xyz[reloc_indices[inside_indices]] = True

    return mask_xyz


@torch.no_grad()
def find_plane_fitting_points(
    mask: torch.Tensor,
    cam: Camera,
    render_pkg: dict,
    opt: OptimizationParams,
    xyz_world: torch.Tensor,
    vis_filter: torch.Tensor,
):

    vis_indices = torch.where(vis_filter)[0]

    if vis_indices.shape[0] < opt.plane_fit_min_points:
        print("Not enough visible points")
        return None

    # Take visible points from the kdtree
    vis_xyz = xyz_world[vis_indices]

    mask = mask > 0.5
    H, W = mask.shape
    vis_pixels = get_image_pixels(vis_xyz, cam)

    vis_pixels = torch.round(vis_pixels).to(torch.int32)
    valid_mask = (
        (vis_pixels[:, 0] >= 0)
        & (vis_pixels[:, 0] < W)
        & (vis_pixels[:, 1] >= 0)
        & (vis_pixels[:, 1] < H)
    )

    valid_vis_pixels = vis_pixels[valid_mask]
    valid_indices = torch.where(valid_mask)[0]  # Get original indices of valid points

    # Check which points are inside the mask
    inside_mask = mask[valid_vis_pixels[:, 1], valid_vis_pixels[:, 0]]
    inside_indices = valid_indices[inside_mask]  # Indices of points inside the mask

    if inside_indices.shape[0] < opt.plane_fit_min_points:
        return None

    close_xyz = vis_xyz[inside_indices]

    neighbour_xyz = close_xyz
    neighbour_indices = vis_indices[inside_indices]

    if neighbour_indices.shape[0] < opt.plane_fit_min_points:
        return None

    upsampled_xyz = neighbour_xyz
    upsampled_indices = neighbour_indices

    # Filter points based on depth
    depth = render_pkg["depth"].squeeze(0)
    valid_depth_mask = (depth > 0) & mask

    depth_xyz = get_xyz_depth(depth, cam)
    depth_xyz = depth_xyz[valid_depth_mask.reshape(-1)]

    # Filter out points that are in out of the bounds of the depth image
    offset = 0.1  # Offset to avoid depth bias
    depth_filter = (
        (upsampled_xyz[:, 0] > depth_xyz[:, 0].min() - offset)
        & (upsampled_xyz[:, 1] > depth_xyz[:, 1].min() - offset)
        & (upsampled_xyz[:, 2] > depth_xyz[:, 2].min() - offset)
        & (upsampled_xyz[:, 0] < depth_xyz[:, 0].max() + offset)
        & (upsampled_xyz[:, 1] < depth_xyz[:, 1].max() + offset)
        & (upsampled_xyz[:, 2] < depth_xyz[:, 2].max() + offset)
    )

    upsampled_indices = upsampled_indices[depth_filter]
    upsampled_xyz = upsampled_xyz[depth_filter]

    if upsampled_xyz.shape[0] < opt.plane_fit_min_points:
        return None

    return upsampled_xyz, upsampled_indices


@torch.no_grad()
def plane_fitting_pipeline(
    mask: torch.Tensor,
    cam: Camera,
    render_pkg: dict,
    opt: OptimizationParams,
    xyz: torch.Tensor,
    vis_filter: torch.Tensor,
):

    plane_points = find_plane_fitting_points(
        mask, cam, render_pkg, opt, xyz, vis_filter
    )

    if plane_points is None:
        return None

    upsampled_xyz, upsampled_indices = plane_points

    # Fit a plane to the upsampled points
    plane_model, inliers, distances = ransac(
        upsampled_xyz, opt.plane_fit_threshold, max_iter=1000
    )
    a, b, c, d = plane_model

    plane_indices = upsampled_indices[inliers].unique()
    num_gaussians = plane_indices.shape[0]

    median_distance = distances.median().item()
    inliers_percentage = inliers.sum().item() / upsampled_xyz.shape[0]

    if median_distance > opt.plane_reject_threshold:
        return None

    translation = torch.mean(upsampled_xyz[inliers], dim=0)
    normal = torch.tensor([a, b, c], device="cuda", dtype=torch.float32)
    normal = normal / torch.norm(normal)

    rotation = normal_to_quaternion(normal)

    print(
        f"Plane {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0 ACCEPTED ({num_gaussians} Gaussians)"
    )
    print(f"Median residual: {median_distance:.4f}")
    print(f"Inliers percentage: {inliers_percentage:.4f}")
    print()

    return (
        rotation,
        translation,
        plane_indices,
    )
