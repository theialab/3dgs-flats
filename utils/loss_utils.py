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

from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def extract_patches(depth, patch_size):
    """Extract (patch_size x patch_size) patches centered at every pixel."""
    pad = patch_size // 2
    depth_padded = F.pad(
        depth.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="replicate"
    )
    patches = F.unfold(depth_padded, kernel_size=patch_size)  # Shape: (1, K*K, H*W)
    return patches.squeeze(0)


def shift_patches(depth, patch_size, shift_h, shift_w):
    """Shift depth map in a specific direction without wrapping."""
    pad = patch_size // 2
    H, W = depth.shape
    depth_padded = F.pad(
        depth.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="replicate"
    )

    # Crop the shifted area instead of using roll()
    start_h = pad + shift_h
    end_h = start_h + H
    start_w = pad + shift_w
    end_w = start_w + W

    depth_shifted = (
        depth_padded[:, :, start_h:end_h, start_w:end_w].squeeze(0).squeeze(0)
    )
    return extract_patches(depth_shifted, patch_size)


def depthtv_loss(depth, patch_size, sample_size=4096):
    patches = extract_patches(depth, patch_size)
    patches_down = shift_patches(depth, patch_size, shift_h=1, shift_w=0)
    patches_right = shift_patches(depth, patch_size, shift_h=0, shift_w=1)

    sample_index = torch.randint(0, patches.size(1), (sample_size,))
    # Compute the L1 loss between the sampled patches and their shifted versions
    loss = l1_loss(patches[:, sample_index], patches_down[:, sample_index]) + l1_loss(
        patches[:, sample_index], patches_right[:, sample_index]
    )

    return loss
