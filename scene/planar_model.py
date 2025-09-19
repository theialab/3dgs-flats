import json
import os

import numpy as np
import torch
import torch.nn as nn
from arguments import OptimizationParams
from plyfile import PlyData, PlyElement
from scene.gaussian_model import GaussianModel
from torch_kdtree import build_kd_tree
from utils.general_utils import get_expon_lr_func
from utils.plane_utils import (
    probability_erf,
    quaternion_to_normal,
    quaternion_to_rotation_matrix,
    zero_z_scale,
    zero_z_coordinate,
    zero_z_tilt,
    rotation_planes_to_world_batch,
    rotation_world_to_planes_batch,
    xyz_planes_to_world_batch,
    xyz_world_to_planes,
    xyz_world_to_planes_batch,
)

NON_PLANAR_ID = -1


class Plane:
    def __init__(
        self,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        start_iteration: int,
    ):
        self._rotation = torch.nn.Parameter(rotation, requires_grad=True)
        self._translation = torch.nn.Parameter(translation, requires_grad=True)
        self.start_iteration = start_iteration

    @property
    def get_rotation(self):
        return nn.functional.normalize(self._rotation.unsqueeze(0)).squeeze(0)

    @property
    def get_normal(self):
        return quaternion_to_normal(self.get_rotation)

    @property
    def get_translation(self):
        return self._translation


class GaussianModelPlanes(GaussianModel):
    """
    A class for a mixture of planes and gaussians
    """

    def __init__(
        self,
        sh_degree: int = 0,
    ):
        super().__init__(sh_degree=sh_degree)

        self.planes = []
        self.plane_ids = torch.empty(0, dtype=torch.long)
        self._planes_q = torch.empty(0)
        self._planes_R = torch.empty(0)
        self._planes_t = torch.empty(0)
        self.plane_optimizer = None

    @property
    def planar_mask(self):
        return self.plane_ids != NON_PLANAR_ID

    @property
    def planar_ids(self):
        return self.plane_ids[self.planar_mask]

    def planar_check(self):
        return self.planar_mask.any()

    @property
    def get_xyz(self):
        xyz = super().get_xyz

        if not self.planar_check():
            return xyz

        # retain z only for non-planar gaussians
        xyz_planar = zero_z_coordinate(xyz[self.planar_mask])
        xyz_planar = xyz_planes_to_world_batch(
            xyz_planar,
            self._planes_R,
            self._planes_t,
        )

        xyz_planar_expanded = torch.empty_like(xyz)
        xyz_planar_expanded[self.planar_mask] = xyz_planar
        xyz = torch.where(self.planar_mask.unsqueeze(1), xyz_planar_expanded, xyz)
        return xyz

    @property
    def get_rotation(self):
        rotation = super().get_rotation

        if not self.planar_check():
            return rotation

        rotation_planar = zero_z_tilt(rotation[self.planar_mask])
        rotation_planar = rotation_planes_to_world_batch(
            rotation_planar, self._planes_q
        )

        rotation_planar_expanded = torch.empty_like(rotation)
        rotation_planar_expanded[self.planar_mask] = rotation_planar
        rotation = torch.where(
            self.planar_mask.unsqueeze(1), rotation_planar_expanded, rotation
        )

        return rotation

    @property
    def get_scaling(self):
        scaling = super().get_scaling

        if not self.planar_check():
            return scaling

        # R is from gaussian to world or to plane
        R = quaternion_to_rotation_matrix(
            self.rotation_activation(self._rotation[self.planar_mask])
        )
        scaling_planar = zero_z_scale(scaling[self.planar_mask], R)

        scaling_planar_expanded = torch.empty_like(scaling)
        scaling_planar_expanded[self.planar_mask] = scaling_planar
        scaling = torch.where(
            self.planar_mask.unsqueeze(1), scaling_planar_expanded, scaling
        )

        return scaling

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation
        )

    def update_planes_params(self):
        if not self.planar_check():
            self._planes_t = torch.empty(0)
            self._planes_q = torch.empty(0)
            self._planes_R = torch.empty(0)

        else:
            self._planes_t = torch.stack(
                [plane.get_translation for plane in self.planes]
            )[self.planar_ids]
            self._planes_q = torch.stack([plane.get_rotation for plane in self.planes])[
                self.planar_ids
            ]
            self._planes_R = quaternion_to_rotation_matrix(self._planes_q)

    def _project_world_to_planes(
        self, xyz_world: torch.Tensor, rotation_world: torch.Tensor
    ):
        planar_mask = self.planar_mask

        xyz = xyz_world.clone()
        xyz[planar_mask] = xyz_world_to_planes_batch(
            xyz_world[planar_mask],
            self._planes_R,
            self._planes_t,
        )

        xyz[planar_mask] = zero_z_coordinate(xyz[planar_mask])
        optimizable_tensors = self.replace_tensor_to_optimizer(xyz, "xyz")
        self._xyz = optimizable_tensors["xyz"]

        rotation = rotation_world.clone()
        rotation[planar_mask] = zero_z_tilt(rotation[planar_mask])
        rotation[planar_mask] = rotation_world_to_planes_batch(
            rotation_world[planar_mask], self._planes_q
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(rotation, "rotation")
        self._rotation = optimizable_tensors["rotation"]

    def training_setup(self, training_args):
        super().training_setup(training_args)

        # everything is non-planar at the beginning
        self.plane_ids = NON_PLANAR_ID * torch.ones(
            self._xyz.shape[0], dtype=torch.long, device="cuda"
        )

        self.plane_rotation_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.plane_rotation_lr_init,
            lr_final=training_args.plane_rotation_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.plane_opt_until_iter,
        )

        self.plane_translation_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.plane_translation_lr_init,
            lr_final=training_args.plane_translation_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.plane_opt_until_iter,
        )

    def update_plane_lr(self, iteration):
        for param_group in self.plane_optimizer.param_groups:
            if "rotation" in param_group["name"]:
                param_group["lr"] = self.plane_rotation_lr_scheduler(
                    iteration - param_group["start_iteration"]
                )
            if "translation" in param_group["name"]:
                param_group["lr"] = self.plane_translation_lr_scheduler(
                    iteration - param_group["start_iteration"]
                )

    def replace_tensor_to_plane_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.plane_optimizer.param_groups:
            if group["name"] == name:

                new_param = nn.Parameter(tensor.requires_grad_(True))
                stored_state = self.plane_optimizer.state.get(group["params"][0], None)

                if stored_state is not None:

                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.plane_optimizer.state[group["params"][0]]

                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.plane_optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = new_param

        return optimizable_tensors

    def add_plane(
        self,
        plane: Plane,
        points_idx: torch.Tensor,
        training_args: OptimizationParams,
    ):

        xyz_world = self.get_xyz
        rotation_world = self.get_rotation

        other_plane_found = False
        for other_plane_id, other_plane in enumerate(self.planes):

            cos_dist = torch.abs(torch.dot(plane.get_normal, other_plane.get_normal))

            d_plane = -torch.dot(plane.get_translation, plane.get_normal)
            d_other_plane = -torch.dot(
                other_plane.get_translation, other_plane.get_normal
            )

            if cos_dist > 0.99 and torch.abs(d_plane - d_other_plane) < 0.1:
                print(f"Plane {other_plane_id} is similar to new plane, merging them")

                other_plane_found = True
                plane_id = other_plane_id
                break

        if not other_plane_found:
            plane_id = len(self.planes)
            self.planes.append(plane)

        # assign plane id to points
        self.plane_ids[points_idx] = plane_id
        self.update_planes_params()
        self._project_world_to_planes(xyz_world, rotation_world)

        if not other_plane_found:
            if not self.plane_optimizer:
                params = []
                params.append(
                    {
                        "params": [plane._rotation],
                        "lr": training_args.plane_rotation_lr_init,
                        "name": f"plane_rotation_{plane_id}",
                        "start_iteration": plane.start_iteration,
                    }
                )
                params.append(
                    {
                        "params": [plane._translation],
                        "lr": training_args.plane_translation_lr_init,
                        "name": f"plane_translation_{plane_id}",
                        "start_iteration": plane.start_iteration,
                    }
                )
                self.plane_optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

            else:
                self.plane_optimizer.add_param_group(
                    {
                        "params": [plane._rotation],
                        "lr": training_args.plane_rotation_lr_init,
                        "name": f"plane_rotation_{plane_id}",
                        "start_iteration": plane.start_iteration,
                    }
                )
                self.plane_optimizer.add_param_group(
                    {
                        "params": [plane._translation],
                        "lr": training_args.plane_translation_lr_init,
                        "name": f"plane_translation_{plane_id}",
                        "start_iteration": plane.start_iteration,
                    }
                )

        return plane_id

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            [
                (
                    plane._rotation,
                    plane._translation,
                    plane.start_iteration,
                )
                for plane in self.planes
            ],
            self.plane_ids,
            self.plane_optimizer.state_dict() if self.plane_optimizer else None,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            optimizer_state_dict,
            self.spatial_lr_scale,
            plane_params,
            plane_ids,
            plane_optimizer_state_dict,
        ) = model_args

        self.training_setup(training_args)
        self.optimizer.load_state_dict(optimizer_state_dict)

        self.plane_ids = plane_ids  # Load after training setup
        self.planes = []

        params = []
        for plane_id, param in enumerate(plane_params):
            plane = Plane(param[0], param[1], start_iteration=param[2])
            self.planes.append(plane)

            params.append(
                {
                    "params": [plane._rotation],
                    "lr": training_args.plane_rotation_lr_init,
                    "name": f"plane_rotation_{plane_id}",
                    "start_iteration": plane.start_iteration,
                }
            )
            params.append(
                {
                    "params": [plane._translation],
                    "lr": training_args.plane_translation_lr_init,
                    "name": f"plane_translation_{plane_id}",
                    "start_iteration": plane.start_iteration,
                }
            )
            self.plane_optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

            if plane_optimizer_state_dict is not None:
                self.plane_optimizer.load_state_dict(plane_optimizer_state_dict)

            self.update_planes_params()

    def save_ply(self, path):
        dir = os.path.dirname(path)
        super().save_ply(os.path.join(dir, "point_cloud_planar.ply"))

        # save planar coordinates for processing
        planes_info = {}
        planes_info["planes"] = {}
        for idx, plane in enumerate(self.planes):
            planes_info["planes"][idx] = {
                "rotation": plane.get_rotation.detach().cpu().numpy().tolist(),
                "translation": plane.get_translation.detach().cpu().numpy().tolist(),
                "start_iteration": plane.start_iteration,
            }
        planes_info["plane_ids"] = (
            self.plane_ids.detach().cpu().numpy().astype(np.int32).tolist()
        )
        with open(os.path.join(dir, "planes.json"), "w") as f:
            json.dump(planes_info, f)

        # save in world coordinates (fused)
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()

        # save in world coordinates (fused)
        scale = self.get_scaling.detach().cpu().numpy()
        rotation = self.get_rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(os.path.join(dir, "point_cloud.ply"))

    def load_ply(self, path):
        dir = os.path.dirname(path)
        planar_path = os.path.join(dir, "point_cloud_planar.ply")

        if not os.path.exists(planar_path):
            super().load_ply(path)

            self.plane_ids = NON_PLANAR_ID * torch.ones(
                self._xyz.shape[0], dtype=torch.long, device="cuda"
            )
            return

        super().load_ply(planar_path)

        with open(os.path.join(dir, "planes.json"), "r") as f:
            planes_info = json.load(f)

        if len(planes_info["planes"]) == 0:
            self.plane_ids = NON_PLANAR_ID * torch.ones(
                self._xyz.shape[0], dtype=torch.long, device="cuda"
            )
            return

        self.plane_ids = torch.tensor(
            planes_info["plane_ids"], dtype=torch.long, device="cuda"
        )
        self.planes = []
        for _, plane_info in planes_info["planes"].items():
            plane = Plane(
                rotation=torch.tensor(plane_info["rotation"], device="cuda"),
                translation=torch.tensor(plane_info["translation"], device="cuda"),
                start_iteration=plane_info["start_iteration"],
            )
            self.planes.append(plane)

        self.update_planes_params()

    @torch.no_grad()
    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )

        (
            self._xyz[dead_indices],
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)
        self.plane_ids[dead_indices] = self.plane_ids[reinit_idx]

        if len(self.planes) > 0:
            self.update_planes_params()

    @torch.no_grad()
    def add_new_gs(self, cap_max: int):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            reset_params=False,
        )
        self.replace_tensors_to_optimizer(inds=add_idx)
        self.plane_ids = torch.cat((self.plane_ids, self.plane_ids[add_idx]))

        if len(self.planes) > 0:
            self.update_planes_params()

    @torch.no_grad()
    def _calculate_plane_distances(self, idx: int, pixel_mask: torch.Tensor):
        xyz_world = self.get_xyz
        plane = self.planes[idx]

        R = quaternion_to_rotation_matrix(plane.get_rotation.unsqueeze(0)).squeeze(0)
        t = plane.get_translation

        xyz_plane = xyz_world_to_planes(xyz_world, R, t)

        non_planar_xy_plane = xyz_plane[pixel_mask][:, :2]
        non_planar_z_plane = xyz_plane[pixel_mask][:, 2]

        planar_xy_plane = xyz_plane[self.plane_ids == idx][:, :2]

        dist_z = torch.abs(non_planar_z_plane)

        kdtree = build_kd_tree(planar_xy_plane)
        dists_xy, _ = kdtree.query(non_planar_xy_plane, nr_nns_searches=1)
        dists_xy = torch.sqrt(dists_xy)  # KDTree returns squared distances

        dist_xy_mean = dists_xy.mean(dim=1)

        return dist_xy_mean.unsqueeze(1), dist_z.unsqueeze(1)

    @torch.no_grad()
    def relocate_to_planar(
        self, idx: int, pixel_mask: torch.Tensor, sigma_res: float, sigma_dist: float
    ):
        if not self.planar_check():
            return

        if not (self.plane_ids == idx).any():
            return

        if not pixel_mask.any():
            return

        # step 1: calculate distances to closest planes together with residuals
        distances, residuals = self._calculate_plane_distances(idx, pixel_mask)

        # step 2: we computer the probabilities
        prob_non_planar_dist = probability_erf(distances, sigma=sigma_dist)
        prob_non_planar_res = probability_erf(residuals, sigma=sigma_res)

        # combine probabilities
        probabilities_planar = (1 - prob_non_planar_res) * (1 - prob_non_planar_dist)
        probabilities_non_planar = 1 - probabilities_planar

        # step 3: high residuals and large distance gaussians have a 1 probability to stay non-planar
        probabilities = torch.cat(
            [
                probabilities_non_planar.sum(dim=1).unsqueeze(1),
                probabilities_planar,
            ],
            dim=1,
        )

        selected_relocation = torch.multinomial(
            probabilities, 1, replacement=False
        ).squeeze()

        selected_plane_ids = torch.where(selected_relocation == 0, NON_PLANAR_ID, idx)

        # step 5: project points to new planes
        xyz_world = self.get_xyz
        rotation_world = self.get_rotation

        self.plane_ids[pixel_mask] = selected_plane_ids
        self.update_planes_params()  # update plane rotations and translations to reflect new plane ids
        self._project_world_to_planes(xyz_world, rotation_world)
