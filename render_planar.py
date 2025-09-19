import os
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path

import mapbox_earcut as earcut
import numpy as np
import open3d as o3d
import torch
import torchvision
from PIL import Image
from rtree import index
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from skimage import measure
from torchvision import transforms
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.planar_model import GaussianModelPlanes
from utils.general_utils import safe_state
from utils.plane_utils import quaternion_to_rotation_matrix
from utils.render_utils import generate_path


def intersection_view_to_world(intersection, view):
    view_to_world_transform = torch.inverse(view.world_view_transform)

    N = intersection.shape[0]
    intersection_h = torch.cat(
        [intersection, torch.ones((N, 1), device=intersection.device)], dim=1
    )

    intersection_world_h = intersection_h @ view_to_world_transform
    intersection_world = intersection_world_h[:, :3] / intersection_world_h[:, 3:4]
    return intersection_world


def create_basis_given_normal(normal: torch.Tensor):
    if torch.abs(normal[0]) > torch.abs(normal[1]):
        vector = torch.tensor([0, 1, 0], dtype=normal.dtype, device=normal.device)
    else:
        vector = torch.tensor([1, 0, 0], dtype=normal.dtype, device=normal.device)

    basis1 = torch.cross(normal, vector, dim=0)
    basis1 = basis1 / torch.norm(basis1)

    basis2 = torch.cross(normal, basis1, dim=0)
    basis2 = basis2 / torch.norm(basis2)

    return basis1, basis2


def get_plane_view(view, plane):
    translation = plane.get_translation
    normal = plane.get_normal

    translation_view = (
        (translation - view.camera_center).unsqueeze(0)
        @ view.world_view_transform[:3, :3]
    ).squeeze(0)
    normal_view = (normal.unsqueeze(0) @ view.world_view_transform[:3, :3]).squeeze(0)

    D = -torch.dot(translation_view, normal_view)
    parameters = torch.tensor(
        [normal_view[0], normal_view[1], normal_view[2], D],
        device="cuda",
    )
    return parameters


def get_rays(view):
    h, w = view.image_height, view.image_width

    y = torch.linspace(0, h - 1, h, device="cuda") - view.Cy
    x = torch.linspace(0, w - 1, w, device="cuda") - view.Cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    rays = torch.stack([xx / view.Fx, yy / view.Fy, torch.ones_like(xx)], dim=-1)
    rays = rays / torch.norm(rays, dim=-1, keepdim=True)
    return rays


def get_ray_plane_intersection(parameters, rays):
    A = parameters[0]
    B = parameters[1]
    C = parameters[2]
    D = parameters[3]

    t = -D / (A * rays[:, :, 0] + B * rays[:, :, 1] + C * rays[:, :, 2])

    intersection = rays * t.unsqueeze(-1)

    return intersection


def project_to_plane_2d(points_3d, origin, R):
    points_local = points_3d - origin
    R_inverse = R.transpose(0, 1)
    points_2d = torch.matmul(R_inverse, points_local.unsqueeze(-1)).squeeze(-1)
    points_2d = points_2d[:, :2]
    return points_2d


def checkerboard_colors(xy_coord, scale):
    omega = 2 * np.pi / scale
    x = xy_coord[:, :, 0]
    y = xy_coord[:, :, 1]

    x_inf = torch.where(
        x == torch.inf, torch.tensor(0.0, device="cuda", dtype=torch.float32), x
    )
    y_inf = torch.where(
        y == torch.inf, torch.tensor(0.0, device="cuda", dtype=torch.float32), y
    )

    sin_y = torch.sin(omega * y_inf)
    sin_x = torch.sin(omega * x_inf)

    colors = torch.where(
        ((sin_x * sin_y) < 0).unsqueeze(-1).repeat(1, 1, 3),
        torch.tensor([0.4, 0.4, 0.4], device="cuda", dtype=torch.float32),
        torch.tensor([0.8, 0.8, 0.8], device="cuda", dtype=torch.float32),
    )
    colors = torch.where(
        (x == torch.inf).unsqueeze(-1).repeat(1, 1, 3),
        torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32),
        colors,
    )
    return colors.permute(2, 0, 1)


def render_set_planar(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
):
    render_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_planes"
    )
    masks_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_mask"
    )
    makedirs(render_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)

    plane_basis = []
    for plane in gaussians.planes:
        normal = plane.get_normal
        basis1, basis2 = create_basis_given_normal(normal)

        plane_basis.append((basis1, basis2))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_pkg = render(view, gaussians, pipeline, background)
        img = render_pkg["render"]
        vis_filter = render_pkg["visibility_filter"].squeeze(-1)

        for idx_plane, plane in enumerate(gaussians.planes):

            plane_condition = gaussians.plane_ids == idx_plane

            if (vis_filter & plane_condition).sum() == 0:
                continue

            basis1, basis2 = plane_basis[idx_plane]

            R = quaternion_to_rotation_matrix(plane.get_rotation)
            plane_origin = plane.get_translation

            parameters = get_plane_view(view, plane)
            rays = get_rays(view)
            intersection = get_ray_plane_intersection(parameters, rays)

            H, W = intersection.shape[0], intersection.shape[1]

            intersection_world = intersection_view_to_world(
                intersection.reshape(-1, 3), view
            )
            intersection_plane_2d = project_to_plane_2d(
                intersection_world, plane_origin, R
            )

            if intersection is not None:
                basis1_view = basis1 @ view.world_view_transform[:3, :3]
                basis2_view = basis2 @ view.world_view_transform[:3, :3]

                translation_camera = (
                    plane.get_translation.repeat(
                        intersection.shape[0], intersection.shape[1], 1
                    )
                    @ view.world_view_transform[:3, :3]
                )

                x = torch.sum(
                    (intersection + translation_camera)
                    * basis1_view.repeat(
                        intersection.shape[0], intersection.shape[1], 1
                    ),
                    dim=-1,
                )
                y = torch.sum(
                    (intersection + translation_camera)
                    * basis2_view.repeat(
                        intersection.shape[0], intersection.shape[1], 1
                    ),
                    dim=-1,
                )

                p = torch.stack([x, y], dim=-1)
                p[intersection[:, :, 2] < 0] = torch.tensor(
                    [torch.inf, torch.inf], device="cuda", dtype=torch.float32
                )

                checkerboard_texture = checkerboard_colors(
                    xy_coord=intersection_plane_2d.reshape(H, W, 2),
                    scale=0.3,
                )

                override_color = torch.where(
                    plane_condition.unsqueeze(1),
                    torch.tensor([1.0, 1.0, 1.0], device="cuda", dtype=torch.float32),
                    torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32),
                ).unsqueeze(0)

                mask = render(
                    view, gaussians, pipeline, background, override_color=override_color
                )["render"]

                img = mask * checkerboard_texture + (1 - mask) * img

                makedirs(os.path.join(masks_path, str(idx_plane)), exist_ok=True)
                torchvision.utils.save_image(
                    mask,
                    os.path.join(masks_path, str(idx_plane), f"{idx:05d}.png"),
                )

        img_final = img

        torchvision.utils.save_image(
            img_final, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )


def voxel_downsample(points, voxel_size=0.01):
    voxel_indices = np.floor(points / voxel_size).astype(int)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    downsampled_points = points[unique_indices]
    return downsampled_points


def remove_statistical_outliers(points, nb_neighbors=20, std_ratio=2.0):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors)
    mean_distances = distances.mean(axis=1)
    std_distances = distances.std(axis=1)
    is_outlier = distances[:, -1] > mean_distances + std_ratio * std_distances
    inlier_points = points[~is_outlier]
    return inlier_points


def triangulate(points2d, grid_resolution):

    try:
        grid_min = points2d.min(axis=0) - grid_resolution
        grid_max = points2d.max(axis=0) + grid_resolution
        grid_shape = np.ceil((grid_max - grid_min) / grid_resolution).astype(int)

        # Create the grid (initialized to 0)
        grid = np.zeros((grid_shape[0], grid_shape[1]), dtype=bool)
        grid_indices = np.floor((points2d - grid_min) / grid_resolution).astype(int)

        valid_mask = (
            (grid_indices[:, 0] >= 0)
            & (grid_indices[:, 0] < grid_shape[0])
            & (grid_indices[:, 1] >= 0)
            & (grid_indices[:, 1] < grid_shape[1])
        )

        grid[grid_indices[valid_mask, 0], grid_indices[valid_mask, 1]] = True

        contours = measure.find_contours(grid, 0.5)

        # Convert to (x, y) format and scale back to world coordinates
        contours = [grid_min + c * grid_resolution for c in contours]

    except Exception as e:
        print("Failed to find contours")
        return [], []

    # Sort contours by length (longest are likely outer boundaries)
    contours = sorted(contours, key=len, reverse=True)
    contours = [c for c in contours if len(c) > 100]

    outer_contours = []
    holes_dict = {}

    # Build R-tree for spatial queries
    idx = index.Index()
    contour_polygons = {}

    for i, contour in enumerate(contours):
        poly = Polygon(contour)
        contour_polygons[i] = poly
        idx.insert(i, poly.bounds)

    for i, contour in enumerate(contours):
        poly = contour_polygons[i]

        # Find potential parent contours
        possible_parents = list(idx.intersection(poly.bounds))
        inside_existing = False

        for parent_idx in possible_parents:
            if parent_idx == i:
                continue  # Skip itself

            parent_poly = contour_polygons[parent_idx]
            if parent_poly.contains(poly):  # Check full containment
                holes_dict.setdefault(
                    tuple(map(tuple, contours[parent_idx])), []
                ).append(contour)
                inside_existing = True
                break

        if not inside_existing:
            outer_contours.append(contour)

    vertices_all = []
    indices_all = []

    for outer in outer_contours:
        holes = holes_dict.get(tuple(map(tuple, outer)), [])

        if len(outer) < 3:
            continue

        vertices = np.vstack([outer] + holes)

        hole_indices = [len(outer)]
        for hole in holes:
            hole_indices.append(hole_indices[-1] + len(hole))

        hole_indices = np.array(hole_indices, dtype=np.uint32)
        hole_indices[-1] = len(vertices)

        try:
            indices = earcut.triangulate_float64(vertices, hole_indices)

            vertices_all.append(vertices)
            indices_all.append(indices)
        except Exception as e:
            print("Failed to triangulate contour")
            continue

    return vertices_all, indices_all


def create_checkerboard_image(size=1000, num_squares=10):
    img = Image.new("RGB", (size, size), color=(204, 204, 204))
    pixels = img.load()
    sq = size // num_squares
    for i in range(size):
        for j in range(size):
            if ((i // sq) + (j // sq)) % 2 == 0:
                pixels[i, j] = (102, 102, 102)

    return img


def planar_mesh(
    gaussians: GaussianModelPlanes,
    scene: Scene,
    device: torch.device,
    grid_resolution: float = 0.1,
    tile_size: float = 5.0,
):
    cameras = scene.getTrainCameras()

    print(f"number of cameras: ", len(cameras))
    print(f"number of planes: ", len(gaussians.planes))

    meshes = []
    for plane_id, plane in enumerate(gaussians.planes):

        origin_plane = plane.get_translation
        R_plane = quaternion_to_rotation_matrix(plane.get_rotation)

        # Initialize a list for storing projected points
        all_projected_points_2d = []

        # Load and project masks
        plane_masks = (
            Path(scene.model_path)
            / "train"
            / "ours_30000"
            / "renders_mask"
            / str(plane_id)
        )
        mask_paths = list(plane_masks.glob("*.png"))
        print(f"plane {plane_id} number of masks: ", len(mask_paths))

        for cam_id, camera in enumerate(cameras):
            mask_path = plane_masks / f"{cam_id:05d}.png"
            if not mask_path.exists():
                continue

            # Load mask
            mask = Image.open(mask_path).convert("L")
            mask = transforms.ToTensor()(mask).to(device)

            """ intersections in camera space """
            parameters = get_plane_view(camera, plane)
            _, _, _, D = parameters

            if torch.abs(D) < 1e-5:  # D close to 0
                continue

            rays = get_rays(camera)
            intersection_points = get_ray_plane_intersection(parameters, rays)
            intersection_points = intersection_points.reshape(-1, 3)

            """ filter out points that are not in the mask """
            mask_indices = mask > 0.8
            if mask_indices.sum() == 0:
                continue

            valid_intersection_points = intersection_points[mask_indices.reshape(-1)]
            valid_intersection_points_w = intersection_view_to_world(
                valid_intersection_points, camera
            )
            """ project 3d points to 2d plane coordinates """
            projected_points_2d = project_to_plane_2d(
                valid_intersection_points_w, origin_plane, R_plane
            )

            projected_points_2d = projected_points_2d.cpu().numpy()
            all_projected_points_2d.append(projected_points_2d)

        # Merge all projected points from all masks
        if len(all_projected_points_2d) == 0:
            print(f"no valid points for plane: ", plane_id)
            continue

        # Compute grid bounds
        all_projected_points_2d = np.concatenate(all_projected_points_2d, axis=0)

        all_projected_points_2d = voxel_downsample(
            all_projected_points_2d, voxel_size=0.01
        )
        all_projected_points_2d = remove_statistical_outliers(
            all_projected_points_2d, nb_neighbors=20, std_ratio=2.0
        )
        vertices_all, triangles_all = triangulate(
            all_projected_points_2d, grid_resolution
        )

        if len(vertices_all) == 0 and len(triangles_all) == 0:
            print(f"no valid triangles for plane: ", plane_id)
            continue

        for vertices, triangles in zip(vertices_all, triangles_all):

            mesh = o3d.geometry.TriangleMesh()

            vertices2d = vertices
            vertices2d = np.concatenate(
                [vertices2d, np.zeros((len(vertices), 1))], axis=1
            )

            vertices3d = (
                np.matmul(
                    R_plane.cpu().numpy()[None, ...], vertices2d[..., None]
                ).squeeze(-1)
                + origin_plane.cpu().numpy()
            )
            vertices3d = np.ascontiguousarray(vertices3d)
            triangles = np.ascontiguousarray(np.array(triangles).reshape(-1, 3))

            uv2d = vertices2d[:, :2]
            uvs = uv2d / tile_size
            triangle_uvs = uvs[triangles].reshape(-1, 2)

            mesh.vertices = o3d.utility.Vector3dVector(vertices3d)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
            mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(triangles))

            meshes.append(mesh)

    mesh_final = meshes[0]
    for mesh in meshes[1:]:
        mesh_final += mesh

    ## ASSIGN THE SAME MATERIAL
    checker_img_path = os.path.join(scene.model_path, "planar_mesh_0.png")
    create_checkerboard_image(1000, 10).save(checker_img_path)
    mesh_final.textures = [o3d.io.read_image(checker_img_path)]

    mesh_path = os.path.join(scene.model_path, "planar_mesh.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh_final)
    print("planar mesh saved at {}".format(mesh_path))


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    skip_video: bool,
    skip_mesh: bool,
    grid_resolution: float,
    tile_size: float,
):
    with torch.no_grad():

        gaussians = GaussianModelPlanes(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set_planar(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            render_set_planar(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_video:
            print("render videos ...")
            traj_dir = os.path.join(
                args.model_path, "traj", "ours_{}".format(scene.loaded_iter)
            )
            os.makedirs(traj_dir, exist_ok=True)
            n_frames = 240
            cam_traj = generate_path(
                scene.getTrainCameras(), n_frames=n_frames, path_type="ellipse"
            )
            render_set_planar(
                dataset.model_path,
                "traj",
                scene.loaded_iter,
                cam_traj,
                gaussians,
                pipeline,
                background,
            )

        if not skip_mesh:
            planar_mesh(
                gaussians,
                scene,
                device=torch.device("cuda"),
                grid_resolution=grid_resolution,
                tile_size=tile_size,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--grid_resolution", default=0.1, type=float)
    parser.add_argument("--tile_size", default=5.0, type=float)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
        args.skip_mesh,
        args.grid_resolution,
        args.tile_size,
    )
