#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser
from os import makedirs

import cv2
import numpy as np
import open3d as o3d
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.planar_model import GaussianModelPlanes
from utils.general_utils import safe_state
from utils.render_utils import create_videos, generate_path


def to_cam_open3d(viewpoint_cam):

    W = viewpoint_cam.image_width
    H = viewpoint_cam.image_height
    ndc2pix = (
        torch.tensor(
            [[W / 2, 0, 0, (W - 1) / 2], [0, H / 2, 0, (H - 1) / 2], [0, 0, 0, 1]]
        )
        .float()
        .cuda()
        .T
    )
    intrins = (viewpoint_cam.projection_matrix @ ndc2pix)[:3, :3].T
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=viewpoint_cam.image_width,
        height=viewpoint_cam.image_height,
        cx=intrins[0, 2].item(),
        cy=intrins[1, 2].item(),
        fx=intrins[0, 0].item(),
        fy=intrins[1, 1].item(),
    )

    extrinsic = (viewpoint_cam.world_view_transform.T).cpu().numpy()
    extrinsic = np.ascontiguousarray(extrinsic.astype(np.float64))

    camera = o3d.camera.PinholeCameraParameters()

    camera.intrinsic = intrinsic
    camera.extrinsic = extrinsic

    return camera


def extract_mesh_bounded(
    scene,
    gaussians,
    pipeline,
    background,
    voxel_size=0.004,
    sdf_trunc=0.02,
    depth_trunc=15.0,
):

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    viewpoint_stack = scene.getTrainCameras()
    for idx, cam in tqdm(enumerate(viewpoint_stack), desc="TSDF integration progress"):

        result = render(cam, gaussians, pipeline, background)

        rgb = result["render"].cpu()
        depth = result["depth"].cpu()
        depth = depth.squeeze(0)

        cam_o3d = to_cam_open3d(cam)

        # make open3d rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(
                np.asarray(
                    np.clip(rgb.permute(1, 2, 0).numpy(), 0.0, 1.0) * 255,
                    order="C",
                    dtype=np.uint8,
                )
            ),
            o3d.geometry.Image(np.asarray(depth.numpy(), order="C")),
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )

        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

    mesh = volume.extract_triangle_mesh()
    return mesh


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print(
        "post processing the mesh to have {} clusterscluster_to_kep".format(
            cluster_to_keep
        )
    )
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        result = render(view, gaussians, pipeline, background)
        rendering = result["render"]
        gt = view.original_image[0:3, :, :]

        depth = result["depth"]
        depth = depth.permute(1, 2, 0)
        depth = (depth * 1000).detach().cpu().numpy().astype(np.uint16)

        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )
        cv2.imwrite(os.path.join(depth_path, "{0:05d}".format(idx) + ".png"), depth)


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    skip_video: bool,
    skip_mesh: bool,
    path_type: str,
):
    with torch.no_grad():

        gaussians = GaussianModelPlanes(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            render_set(
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
                scene.getTrainCameras(),
                n_frames=n_frames,
                path_type=path_type,
            )
            render_set(
                dataset.model_path,
                "traj",
                scene.loaded_iter,
                cam_traj,
                gaussians,
                pipeline,
                background,
            )

            create_videos(
                base_dir=traj_dir,
                input_dir=traj_dir,
                out_name="render_traj",
                num_frames=n_frames,
            )

        if not skip_mesh:
            print("extract mesh ...")

            name = "fuse.ply"
            path = os.path.join(args.model_path, name)

            depth_trunc = scene.cameras_extent * 2.0
            voxel_size = depth_trunc / 1024
            sdf_trunc = 5.0 * voxel_size

            print(
                "Extracting mesh with depth_trunc: {}, voxel_size: {}, sdf_trunc: {}".format(
                    depth_trunc, voxel_size, sdf_trunc
                )
            )

            mesh = extract_mesh_bounded(
                scene,
                gaussians,
                pipeline,
                background,
                depth_trunc=depth_trunc,
                voxel_size=voxel_size,
                sdf_trunc=sdf_trunc,
            )

            o3d.io.write_triangle_mesh(path, mesh)
            print("mesh saved at {}".format(path))

            mesh_post = post_process_mesh(mesh, cluster_to_keep=50)

            path_post = os.path.join(args.model_path, name.replace(".ply", "_post.ply"))
            o3d.io.write_triangle_mesh(path_post, mesh_post)
            print("mesh post processed saved at {}".format(path_post))


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
    parser.add_argument(
        "--path_type",
        default="interpolated",
        choices=["interpolated", "ellipse"],
        type=str,
    )
    parser.add_argument("--quiet", action="store_true")
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
        args.path_type,
    )
