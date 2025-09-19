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
import time
from argparse import ArgumentParser

import queue
from joblib import Parallel, delayed

scannetpp_scenes = [
    "0a7cc12c0e",
    "0cf2e9402d",
    "0e75f3c4d9",
    "1ae9e5d2a6",
    "1b75758486",
    "1c4b893630",
    "2e74812d00",
    "4c5c60fa76",
    "4ea827f5a1",
    "5748ce6f01",
    "7079b59642",
]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path")
parser.add_argument("--num_gpus", default=2, type=int)
parser.add_argument("--scannetpp", "-spp", type=str, required=True)

args = parser.parse_args()


all_scenes = []
all_scenes.extend(scannetpp_scenes)

# Put indices in queue
q = queue.Queue(maxsize=args.num_gpus)
for i in range(args.num_gpus):
    q.put(i)


def trainer(cmd):
    gpu = q.get()
    cmd = cmd + f" --ip 127.0.0.{30 + gpu}"

    print(f"Running command on GPU: {gpu}: {cmd}")

    start_time = time.time()
    os.system(f"CUDA_VISIBLE_DEVICES={gpu} {cmd}")
    total_time = (time.time() - start_time) / 60.0

    print(f"Total time: {total_time:.2f} minutes")

    # return gpu id to queue
    q.put(gpu)


def renderer(cmd):
    gpu = q.get()

    print(f"Running command on GPU: {gpu}: {cmd}")
    os.system(f"CUDA_VISIBLE_DEVICES={gpu} {cmd}")

    # return gpu id to queue
    q.put(gpu)


if not args.skip_training:
    print("Populating training commands")

    training_cmds = []

    os.makedirs(args.output_path, exist_ok=True)
    common_args = " --eval --test_iterations -1 --quiet "

    for scene in scannetpp_scenes:

        training_cmds.append(
            "python train_planar.py -s "
            + f" {args.scannetpp}/{scene}/iphone "
            + f" -m {args.output_path}/{scene} "
            + " --init_type sfm "
            + f" --config configs/iphone/{scene}.json "
            + f" --mask_root data_preparation/sam2/output/{scene} "
            + common_args
        )

    print(
        f"Running {len(training_cmds)} training commands in parallel on {args.num_gpus} GPUs"
    )

    Parallel(n_jobs=args.num_gpus, backend="threading")(
        delayed(trainer)(cmd) for cmd in training_cmds
    )

if not args.skip_rendering:
    print("Populating rendering commands")

    rendering_cmds = []

    common_args = " --skip_video --eval "
    for scene in scannetpp_scenes:

        rendering_cmds.append(
            "python render.py --iteration 30000 "
            + f" -m {args.output_path}/{scene} "
            + common_args
        )
        rendering_cmds.append(
            "python render_planar.py --iteration 30000 "
            + f" -m {args.output_path}/{scene} "
            + common_args
        )
    print(f"Running rendering commands in parallel on {args.num_gpus} GPUs")
    Parallel(n_jobs=args.num_gpus, backend="threading")(
        delayed(renderer)(cmd) for cmd in rendering_cmds
    )

if not args.skip_metrics:
    print("Populating metrics commands")

    scenes_string = ""
    for scene in all_scenes:
        scenes_string += f" {args.output_path}/{scene} "

    print("Running metrics computation")
    os.system("python metrics.py " + " -m " + scenes_string)
