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

import glob
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def readDepthImages(renders_dir, gt_dir, depth_scale=1000.0):
    renders = []
    gts = []

    renders_files = sorted(os.listdir(renders_dir))
    gt_files = sorted(os.listdir(gt_dir))

    assert len(renders_files) == len(gt_files)

    for render_fname, gt_fname in zip(renders_files, gt_files):

        render = cv2.imread(renders_dir / render_fname, cv2.IMREAD_UNCHANGED)
        render = render / depth_scale

        gt = cv2.imread(gt_dir / gt_fname, cv2.IMREAD_UNCHANGED)
        gt = gt / depth_scale
        gt = cv2.resize(
            gt, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        mask = gt != 0

        render = render[mask]
        gt = gt[mask]

        renders.append(torch.from_numpy(render).cuda())
        gts.append(torch.from_numpy(gt).cuda())

    return renders, gts


def rmse(renders, gts):
    return torch.sqrt(torch.mean((renders - gts) ** 2))


def mae(renders, gts):
    return torch.mean(torch.abs(renders - gts))


def absrel(renders, gts):
    return torch.mean(torch.abs(renders - gts) / gts)


def threshold_accuracy(renders, gts, thresholds=[1.25, 1.25**2, 1.25**3]):
    delta = torch.maximum(renders / gts, gts / renders)
    accuracies = []
    for tau in thresholds:
        accuracies.append(torch.mean((delta < tau).to(torch.float32)))

    return accuracies


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                if method == "ours_7000":
                    continue

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

                depth_renders_dir = method_dir / "depth"
                depth_gt_dir = method_dir / "depth_gt"

                renders_depth, gts_depth = readDepthImages(
                    depth_renders_dir, depth_gt_dir
                )

                rmses = []
                maes = []
                absrels = []
                thresholds = []

                for idx in tqdm(
                    range(len(gts_depth)), desc="Depth evaluation progress"
                ):
                    rmses.append(rmse(renders_depth[idx], gts_depth[idx]))
                    maes.append(mae(renders_depth[idx], gts_depth[idx]))
                    absrels.append(absrel(renders_depth[idx], gts_depth[idx]))
                    thresholds.append(
                        threshold_accuracy(renders_depth[idx], gts_depth[idx])
                    )

                thresholds = torch.tensor(thresholds)

                print("Novel view synthesis metrics:")
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
                print("")

                print("Depth estimation metrics:")
                print("  RMSE  : {:>12.7f}".format(torch.tensor(rmses).mean()))
                print("  MAE   : {:>12.7f}".format(torch.tensor(maes).mean()))
                print("  AbsRel: {:>12.7f}".format(torch.tensor(absrels).mean()))
                print("  delta<1.25: {:>12.7f}".format(thresholds[:, 0].mean()))
                print("  delta<1.25^2: {:>12.7f}".format(thresholds[:, 1].mean()))
                print("  delta<1.25^3: {:>12.7f}".format(thresholds[:, 2].mean()))
                print("")

                full_dict[scene_dir][method].update(
                    {
                        "SSIM": torch.tensor(ssims).mean().item(),
                        "PSNR": torch.tensor(psnrs).mean().item(),
                        "LPIPS": torch.tensor(lpipss).mean().item(),
                    }
                )

                full_dict[scene_dir][method].update(
                    {
                        "RMSE": torch.tensor(rmses).mean().item(),
                        "MAE": torch.tensor(maes).mean().item(),
                        "AbsRel": torch.tensor(absrels).mean().item(),
                        "delta<1.25": thresholds[:, 0].mean().item(),
                        "delta<1.25^2": thresholds[:, 1].mean().item(),
                        "delta<1.25^3": thresholds[:, 2].mean().item(),
                    }
                )

                per_view_dict[scene_dir][method].update(
                    {
                        "SSIM": {
                            name: ssim
                            for ssim, name in zip(
                                torch.tensor(ssims).tolist(), image_names
                            )
                        },
                        "PSNR": {
                            name: psnr
                            for psnr, name in zip(
                                torch.tensor(psnrs).tolist(), image_names
                            )
                        },
                        "LPIPS": {
                            name: lp
                            for lp, name in zip(
                                torch.tensor(lpipss).tolist(), image_names
                            )
                        },
                    }
                )

                per_view_dict[scene_dir][method].update(
                    {
                        "RMSE": {
                            name: rmse
                            for rmse, name in zip(
                                torch.tensor(rmses).tolist(), image_names
                            )
                        },
                        "MAE": {
                            name: mae
                            for mae, name in zip(
                                torch.tensor(maes).tolist(), image_names
                            )
                        },
                        "AbsRel": {
                            name: absrel
                            for absrel, name in zip(
                                torch.tensor(absrels).tolist(), image_names
                            )
                        },
                        "delta<1.25": {
                            name: delta
                            for delta, name in zip(
                                thresholds[:, 0].tolist(), image_names
                            )
                        },
                        "delta<1.25^2": {
                            name: delta
                            for delta, name in zip(
                                thresholds[:, 1].tolist(), image_names
                            )
                        },
                        "delta<1.25^3": {
                            name: delta
                            for delta, name in zip(
                                thresholds[:, 2].tolist(), image_names
                            )
                        },
                    }
                )

            with open(scene_dir + "/results.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(e)
            print("Unable to compute metrics for model", scene_dir)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    args = parser.parse_args()
    evaluate(args.model_paths)
