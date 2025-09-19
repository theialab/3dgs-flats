import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor_hf
from sam2.sam2_video_predictor import SAM2VideoPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM2 for mask propagation")
    parser.add_argument(
        "--orig_frames_dir",
        "-i",
        type=str,
        help="Path to the original frames directory",
    )
    parser.add_argument(
        "--train_test_json",
        "-t",
        type=str,
        help="Mark frames in json as a test frames.",
    )
    parser.add_argument("--test_k", type=int, help="Sample test frames every")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--planerecnet_dir",
        "-p",
        type=str,
        help="Path to the PlaneRecNet results directory",
    )
    parser.add_argument("--cost_threshold", type=float, default=0.1)
    return parser.parse_args()


def copy_files(src_dir: str, dst_dir: str, train_list: list[str]):
    train_list = set(train_list)

    os.makedirs(dst_dir, exist_ok=True)
    orig_frame_paths = sorted(glob.glob(f"{src_dir}/*"))

    for i, path in enumerate(orig_frame_paths):
        name = Path(path).name
        if name not in train_list:
            continue

        ext = Path(path).suffix
        if ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]:
            os.system(f"cp {path} {os.path.join(dst_dir, f'{i:05d}.JPG')}")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    image_size = Image.open(path).size
    return image_size


def get_points_and_labels_from_mask(mask: np.ndarray):
    positive_coordinates = np.argwhere(mask > 0.5)
    positive_sampled_points = positive_coordinates[
        np.random.choice(len(positive_coordinates), 7, replace=False)
    ]

    negative_coordinates = np.argwhere(mask < 0.5)
    negative_sampled_points = negative_coordinates[
        np.random.choice(len(negative_coordinates), 3, replace=False)
    ]

    positive_sampled_points = np.flip(positive_sampled_points, axis=1)
    negative_sampled_points = np.flip(negative_sampled_points, axis=1)
    points = np.concatenate([positive_sampled_points, negative_sampled_points]).astype(
        np.float32
    )
    labels = np.array(
        [1] * len(positive_sampled_points) + [0] * len(negative_sampled_points),
        np.int32,
    )
    return points, labels


def propagate_masks(
    prev_video_segments: dict,
    out_frame_idx: int,
    predictor: SAM2VideoPredictor,
    inference_state,
    reverse=False,
):
    # run propagation throughout the video and collect the results in a dict
    video_segments = prev_video_segments.copy()

    for (
        other_out_frame_idx,
        other_out_obj_ids,
        other_out_mask_logits,
    ) in predictor.propagate_in_video(
        inference_state, start_frame_idx=out_frame_idx, max_frame_num_to_track=16
    ):
        video_segments[other_out_frame_idx] = {
            out_obj_id: (other_out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(other_out_obj_ids)
        }

    if reverse:
        for (
            other_out_frame_idx,
            other_out_obj_ids,
            other_out_mask_logits,
        ) in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=out_frame_idx,
            reverse=True,
            max_frame_num_to_track=16,
        ):
            video_segments[other_out_frame_idx] = {
                out_obj_id: (other_out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(other_out_obj_ids)
            }

    return video_segments


def add_new_objects(
    video_segments: dict,
    out_frame_idx: int,
    out_obj_id: int,
    masks_to_add: list[np.ndarray],
    predictor,
    inference_state,
):
    for i, mask in enumerate(masks_to_add):
        new_ann_obj_id = out_obj_id + i + 1
        points, labels = get_points_and_labels_from_mask(mask)

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=out_frame_idx,
            obj_id=new_ann_obj_id,
            points=points,
            labels=labels,
        )

    return propagate_masks(video_segments, out_frame_idx, predictor, inference_state)


def prepare_predictor():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    predictor = build_sam2_video_predictor_hf(
        "facebook/sam2.1-hiera-large", device=device
    )
    return predictor


def prepare_inference_state(predictor, orig_frames_dir, train_list, tmp_frames_dir):
    orig_frame_names = sorted(train_list)
    size = copy_files(orig_frames_dir, tmp_frames_dir, train_list=train_list)
    inference_state = predictor.init_state(video_path=tmp_frames_dir)
    return inference_state, orig_frame_names, size


def get_cmap_color(idx):
    cmap = plt.get_cmap("tab20")
    rgba_color = cmap(idx % 20)
    return tuple(int(255 * x) for x in rgba_color[2::-1])


def visualize_masks(orig_frames_dir: str, orig_frame_names: list[str], output_dir: str):

    orig_frame_stems = [Path(name).stem for name in orig_frame_names]

    tmp_masks_dir = "tmp/test_masks"
    os.makedirs(tmp_masks_dir, exist_ok=True)

    images_paths = sorted(glob.glob(f"{orig_frames_dir}/*"))
    num_masks = len(glob.glob(f"{output_dir}/*"))

    for image_index, image_path in tqdm(enumerate(images_paths), "visualizing masks"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        stem = Path(image_path).stem

        if stem not in orig_frame_stems:
            continue

        for i in range(num_masks):

            mask_path = f"{output_dir}/{i}/{stem}.png"

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = mask > 0.5

                image[mask] = np.array(get_cmap_color(i))[:3]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{tmp_masks_dir}/{image_index:05d}.png", image)

    os.system(
        f"ffmpeg -framerate 8 -pattern_type glob -i '{tmp_masks_dir}/*.png' -c:v libx264 -pix_fmt yuv420p '{output_dir}/out.mp4' -y"
    )
    shutil.rmtree(tmp_masks_dir)


def main(args):

    cost_threshold = args.cost_threshold

    planerecnet_dir = args.planerecnet_dir
    output_dir = args.output_dir

    orig_frames_dir = args.orig_frames_dir
    tmp_frames_dir = "tmp/frames"

    shutil.rmtree(tmp_frames_dir, ignore_errors=True)

    if args.train_test_json is not None:
        with open(args.train_test_json, "r") as f:
            train_test_json = json.load(f)

        train_list = train_test_json["train"]
    else:
        train_list = [
            Path(path).name for path in sorted(glob.glob(f"{orig_frames_dir}/*"))
        ]

        if args.test_k > 0:
            train_list = [
                c for idx, c in enumerate(train_list) if idx % args.test_k != 0
            ]

    predictor = prepare_predictor()
    inference_state, orig_frame_names, size = prepare_inference_state(
        predictor, orig_frames_dir, train_list, tmp_frames_dir
    )

    video_segments = {}
    for out_frame_idx in tqdm(
        range(len(orig_frame_names)),
        desc="processing frames and matching masks by IoU",
    ):
        stem = Path(orig_frame_names[out_frame_idx]).stem

        planerecnet_masks = sorted(glob.glob(f"{planerecnet_dir}/{stem}_mask_*.png"))
        if not planerecnet_masks:
            continue

        masks = []
        for mask_path in planerecnet_masks:
            mask = Image.open(mask_path)
            mask = mask.resize(size)
            mask = np.array(mask)
            mask = mask > 0.5
            masks.append(mask)

        if not video_segments:
            # if the previous frame is not available, we add the masks as new objects
            masks_to_add = masks
            out_obj_id = -1

            video_segments = add_new_objects(
                video_segments,
                out_frame_idx,
                out_obj_id,
                masks_to_add,
                predictor,
                inference_state,
            )

        else:
            if out_frame_idx not in video_segments:
                # if we ran out of predicted masks, we propagate the masks a bit further
                video_segments = propagate_masks(
                    video_segments, out_frame_idx, predictor, inference_state
                )

            # if the previous frame is available, we match the masks to the previous frames
            cost_matrix = np.zeros(
                (len(video_segments[out_frame_idx]), len(planerecnet_masks)),
                dtype=np.float32,
            )

            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                out_mask = out_mask[0]
                for mask_id, mask in enumerate(masks):
                    intersection = (out_mask & mask).sum()
                    union = (out_mask | mask).sum()

                    iou = intersection / (union + 1e-6)
                    cost_matrix[out_obj_id, mask_id] = iou

            row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)
            assignments = np.array(list(zip(row_indices, col_indices)))

            masks_to_add = []
            for i, j in assignments:
                if cost_matrix[i, j] < cost_threshold:
                    mask = masks[j]  # add a new object
                    masks_to_add.append(mask)

            if masks_to_add:
                video_segments = add_new_objects(
                    video_segments,
                    out_frame_idx,
                    out_obj_id,
                    masks_to_add,
                    predictor,
                    inference_state,
                )

    # save the segmentation results
    for out_frame_idx in tqdm(range(len(orig_frame_names)), desc="saving masks"):

        if out_frame_idx not in video_segments:
            continue

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            os.makedirs(f"{output_dir}/{out_obj_id}", exist_ok=True)

            out_mask = out_mask[0]

            if out_mask.sum() == 0:
                continue

            out_mask_write = out_mask.astype(np.uint8) * 255
            out_mask_write = cv2.cvtColor(out_mask_write, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(
                f"{output_dir}/{out_obj_id}/{Path(orig_frame_names[out_frame_idx]).stem}.png",
                out_mask_write,
            )

    shutil.rmtree(tmp_frames_dir)
    visualize_masks(orig_frames_dir, orig_frame_names, output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
