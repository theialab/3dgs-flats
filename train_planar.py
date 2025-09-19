import glob
import json
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import Scene
from scene.gaussian_model import build_scaling_rotation
from scene.planar_model import GaussianModelPlanes, Plane
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import depthtv_loss, l1_loss, ssim
from utils.plane_fitting_utils import (
    find_closest_mask_points,
    get_image_pixels,
    plane_fitting_pipeline,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def name_stem(image_name):
    return Path(image_name).stem


def get_camera_image_names(scene: Scene):
    return {name_stem(cam.image_name): cam for cam in scene.getTrainCameras()}


def get_masks_scene(mask_root, stem_to_cam: dict):
    masks_dict = {}
    num_planes = len(glob.glob(f"{mask_root}/*"))
    for i in range(num_planes):
        cam_names = []
        masks = []
        for mask_path in sorted(glob.glob(f"{mask_root}/{i}/*.png")):
            mask_stem = name_stem(mask_path)

            cam = stem_to_cam.get(mask_stem)

            if not cam:
                continue

            # Load mask
            mask = cv2.imread(mask_path)
            mask = mask[:, :, :3]
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (cam.image_width, cam.image_height))
            mask = mask.astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)

            # Take big enough masks
            if (mask > 0.5).sum() > 128 * 128:
                cam_names.append(name_stem(cam.image_name))
                masks.append(mask)

        if len(cam_names) == 0 or len(masks) == 0:
            continue

        masks_dict[i] = {}
        for cam_name, mask in zip(cam_names, masks):
            masks_dict[i][cam_name] = mask

    # Removing planes visible from a few views
    masks_dict = {k: v for k, v in masks_dict.items() if len(v) > 10}
    return masks_dict


def plane_one_hot(plane_ids, num_planes):
    valid_mask = plane_ids >= 0
    one_hot = F.one_hot(plane_ids[valid_mask], num_classes=num_planes).to(torch.float32)
    result = torch.zeros(
        len(plane_ids), num_planes, dtype=torch.float32, device=plane_ids.device
    )
    result[valid_mask] = one_hot
    return result


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    debug_render,
):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModelPlanes(sh_degree=dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=True)

    print("Loading masks...")
    stem_to_cam_dict = get_camera_image_names(scene)
    masks_dict = get_masks_scene(args.mask_root, stem_to_cam_dict)
    num_masks = sum([len(masks_dict[i]) for i in masks_dict])
    print(f"Found {len(masks_dict)} planes, and {num_masks} masks")

    optimized_masks_dict = {}
    plane_to_mask_id = {}

    if num_masks == 0:
        raise ValueError(f"No masks found in root {args.mask_root} ")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussians.training_setup(opt)

    # DEBUG
    if debug_render:
        os.makedirs(f"{scene.model_path}/xy_view", exist_ok=True)
        os.makedirs(f"{scene.model_path}/xz_view", exist_ok=True)
        os.makedirs(f"{scene.model_path}/yz_view", exist_ok=True)
    # END DEBUG

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    depthtv_loss_weight = lambda iteration: (
        opt.depthtv_loss_weight if iteration < opt.plane_opt_until_iter else 0.0
    )
    planar_mask_loss_weight = lambda iteration: (
        opt.planar_mask_loss_weight if iteration < opt.plane_opt_until_iter else 0.0
    )

    def rendering_loss(viewpoint_cam, depthtv_loss_weight=0, planar_mask_loss_weight=0):

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        num_planes = len(gaussians.planes)
        if num_planes > 0 and planar_mask_loss_weight > 0:

            gt_masks = []
            gt_ids = []

            for i in range(num_planes):
                mask_gt = optimized_masks_dict[i].get(
                    name_stem(viewpoint_cam.image_name)
                )
                if mask_gt is None:
                    continue

                gt_masks.append(mask_gt)
                gt_ids.append(i)

            if len(gt_ids) == 0:
                extra_attrs = None

            else:
                extra_attrs = plane_one_hot(gaussians.plane_ids, num_planes)
                extra_attrs = extra_attrs[:, gt_ids]

        else:
            extra_attrs = None

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, extra_attrs=extra_attrs)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()
        loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()

        if depthtv_loss_weight > 0:
            depth = render_pkg["depth"]
            loss = loss + depthtv_loss_weight * depthtv_loss(
                depth.squeeze(0), patch_size=8, sample_size=4096
            )

        if num_planes > 0 and planar_mask_loss_weight > 0 and len(gt_masks) > 0:
            extra = render_pkg["extra"]

            mask_loss = 0
            for i, mask_gt in enumerate(gt_masks):

                mask_gt = mask_gt.cuda()
                mask_pred = extra[i]
                mask_loss += l1_loss(mask_pred, mask_gt)

            mask_loss = mask_loss / len(gt_masks)
            loss = loss + planar_mask_loss_weight * mask_loss

        loss.backward(retain_graph=True)

        return loss, Ll1, render_pkg

    iteration = first_iter
    while iteration <= opt.iterations:

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        xyz_lr = gaussians.update_learning_rate(iteration)

        if gaussians.planar_check():
            gaussians.update_plane_lr(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Optimize planes
        if (
            gaussians.planar_check()
            and iteration not in testing_iterations
            and iteration < opt.plane_opt_until_iter
            and iteration % 100 == 0
        ):
            gaussians.plane_optimizer.zero_grad(set_to_none=True)
            gaussians.optimizer.zero_grad(set_to_none=True)

            for _ in range(opt.plane_opt_iterations):
                # Pick a random Camera
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()

                viewpoint_cam = viewpoint_stack.pop(
                    randint(0, len(viewpoint_stack) - 1)
                )

                loss, Ll1, render_pkg = rendering_loss(
                    viewpoint_cam,
                    depthtv_loss_weight=opt.depthtv_loss_weight,
                    planar_mask_loss_weight=opt.planar_mask_loss_weight,
                )

                gaussians.plane_optimizer.step()

                gaussians.plane_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.zero_grad(set_to_none=True)

                gaussians.update_planes_params()

        # Optimize Gaussians
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        loss, Ll1, render_pkg = rendering_loss(
            viewpoint_cam,
            depthtv_loss_weight(iteration),
            planar_mask_loss_weight(iteration),
        )

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

                with open(
                    f"{scene.model_path}/plane_to_mask_id_{iteration}.json", "w"
                ) as f:
                    json.dump(plane_to_mask_id, f)

            # Plane relocation
            if (
                gaussians.planar_check()
                and iteration not in testing_iterations
                and iteration < opt.plane_opt_until_iter
                and iteration % 100 == 0
            ):

                # visible and not dead gaussians
                vis_filter = render_pkg["visibility_filter"].squeeze(-1)

                for idx in range(len(gaussians.planes)):

                    mask = optimized_masks_dict[idx].get(
                        name_stem(viewpoint_cam.image_name)
                    )

                    if mask is None:
                        continue

                    mask = mask.cuda()

                    reloc_filter = vis_filter & (gaussians.planar_mask != idx)
                    reloc_pixels = get_image_pixels(
                        gaussians.get_xyz[reloc_filter],
                        viewpoint_cam,
                    )
                    pixel_mask = find_closest_mask_points(
                        mask,
                        reloc_pixels,
                        reloc_filter,
                    )
                    gaussians.relocate_to_planar(
                        idx,
                        pixel_mask,
                        sigma_res=opt.plane_sigma_res,
                        sigma_dist=opt.plane_sigma_dist,
                    )

            # Plane fitting
            if (
                len(masks_dict) > 0
                and iteration not in testing_iterations
                and iteration >= opt.plane_fit_iter
                and iteration < opt.plane_fit_until_iter
                and iteration % 100 == 0
            ):

                # Take visible and solid points
                vis_filter = render_pkg["visibility_filter"].squeeze(-1)
                vis_filter = vis_filter & (gaussians.get_opacity.squeeze(-1) > 0.1)

                if not vis_filter.any():
                    continue

                keys = list(masks_dict.keys())
                for k in keys:

                    mask = masks_dict[k].get(name_stem(viewpoint_cam.image_name))

                    if mask is None:
                        continue

                    mask = mask.cuda()
                    plane_params = plane_fitting_pipeline(
                        mask,
                        viewpoint_cam,
                        render_pkg,
                        opt,
                        gaussians.get_xyz,
                        vis_filter,
                    )

                    if plane_params is not None:
                        plane_r, plane_t, plane_indices = plane_params
                        plane = Plane(plane_r, plane_t, iteration)

                        plane_id = gaussians.add_plane(plane, plane_indices, opt)

                        if plane_id not in optimized_masks_dict.keys():
                            plane_to_mask_id[plane_id] = [k]
                            optimized_masks_dict[plane_id] = masks_dict[k].copy()

                        else:
                            plane_to_mask_id[plane_id].append(k)
                            for key in masks_dict[k]:
                                if key not in optimized_masks_dict[plane_id]:
                                    optimized_masks_dict[plane_id][key] = masks_dict[k][
                                        key
                                    ]
                                else:
                                    optimized_masks_dict[plane_id][key] = torch.maximum(
                                        optimized_masks_dict[plane_id][key],
                                        masks_dict[k][key],
                                    )

                        del masks_dict[k]

            # MCMC Densification
            if (
                iteration < opt.densify_until_iter
                and iteration > opt.densify_from_iter
                and iteration % opt.densification_interval == 0
            ):
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)

                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(cap_max=args.cap_max)

            if iteration < opt.iterations:

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                L = build_scaling_rotation(
                    gaussians.get_scaling,
                    gaussians.rotation_activation(gaussians._rotation),
                )
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))

                noise = (
                    torch.randn_like(gaussians._xyz)
                    * (op_sigmoid(1 - gaussians.get_opacity))
                    * args.noise_lr
                    * xyz_lr
                )

                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)

                # Planar points are not affected by noise in z direction
                # noise[gaussians.planar_mask] = 0.0
                gaussians._xyz.add_(noise)

        # DEBUG INFO
        with torch.no_grad():
            if (
                debug_render
                and gaussians.planar_check()
                and (iteration % 1000 == 0 or iteration == 10)
            ):

                min_lim = -9
                max_lim = 9

                plane_ids = gaussians.plane_ids.detach().cpu().numpy()
                xyz_world = gaussians.get_xyz.detach().cpu().numpy()
                alphas = gaussians.get_opacity.detach().cpu().numpy()

                xyz_world = xyz_world[plane_ids != -1]
                colors = plane_ids[plane_ids != -1]
                alphas = alphas[plane_ids != -1]

                fig = plt.figure()

                plt.scatter(
                    xyz_world[:, 0],
                    xyz_world[:, 1],
                    c=colors,
                    alpha=alphas,
                    s=0.1,
                )
                ax = plt.gca()
                ax.set_aspect("equal")
                ax.set_xlim(min_lim, max_lim)
                ax.set_ylim(min_lim, max_lim)
                fig.savefig(f"{scene.model_path}/xy_view/{iteration}.png")
                plt.close()

                fig = plt.figure()

                plt.scatter(
                    xyz_world[:, 0],
                    xyz_world[:, 2],
                    c=colors,
                    alpha=alphas,
                    s=0.1,
                )
                ax = plt.gca()
                ax.set_aspect("equal")
                ax.set_xlim(min_lim, max_lim)
                ax.set_ylim(min_lim, max_lim)
                fig.savefig(f"{scene.model_path}/xz_view/{iteration}.png")
                plt.close()

                fig = plt.figure()

                plt.scatter(
                    xyz_world[:, 1],
                    xyz_world[:, 2],
                    c=colors,
                    alpha=alphas,
                    s=0.1,
                )
                ax = plt.gca()
                ax.set_aspect("equal")
                ax.set_xlim(min_lim, max_lim)
                ax.set_ylim(min_lim, max_lim)
                fig.savefig(f"{scene.model_path}/yz_view/{iteration}.png")
                plt.close()

        # END DEBUG INFO

        if iteration in checkpoint_iterations:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save(
                (gaussians.capture(), iteration),
                scene.model_path + "/chkpnt" + str(iteration) + ".pth",
            )

        iteration += 1


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()

    if iteration == 30000:
        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )


def load_config(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.11")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--mask_root", type=str, default=None)
    parser.add_argument("--debug_render", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])

    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.debug_render,
    )

    # All done
    print("\nTraining complete.")
