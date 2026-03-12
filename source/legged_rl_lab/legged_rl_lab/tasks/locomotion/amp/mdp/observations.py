# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-specific observation functions.

These observations are used as input to the AMP discriminator.
The discriminator needs motion-related features (without command/gravity info)
to distinguish between reference and policy-generated motions.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def amp_joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default for AMP observations.

    Returns joint positions offset by their default values.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def amp_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities for AMP observations."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def amp_base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base linear velocity in base frame for AMP observations."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def amp_base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base angular velocity in base frame for AMP observations."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def amp_foot_positions_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """Foot positions in base frame for AMP observations.

    Returns the positions of specified body frames relative to the base frame.
    """
    asset = env.scene[asset_cfg.name]

    # Resolve body indices
    body_ids, _ = asset.find_bodies(asset_cfg.body_names)

    # Get body positions in world frame
    foot_pos_w = asset.data.body_pos_w[:, body_ids, :]  # (num_envs, num_feet, 3)

    # Get base position and orientation
    base_pos_w = asset.data.root_pos_w  # (num_envs, 3)
    base_quat_w = asset.data.root_quat_w  # (num_envs, 4)

    # Transform to base frame
    # Compute relative position
    rel_pos = foot_pos_w - base_pos_w.unsqueeze(1)  # (num_envs, num_feet, 3)

    # Rotate to base frame using quaternion inverse
    # q_inv = [w, -x, -y, -z]
    quat_inv = base_quat_w.clone()
    quat_inv[:, 1:] *= -1.0

    # Apply rotation to each foot
    num_feet = foot_pos_w.shape[1]
    foot_pos_b = torch.zeros_like(rel_pos)
    for i in range(num_feet):
        foot_pos_b[:, i, :] = _quat_rotate(quat_inv, rel_pos[:, i, :])

    # Flatten: (num_envs, num_feet * 3)
    return foot_pos_b.reshape(foot_pos_b.shape[0], -1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (wxyz format).

    Args:
        q: Quaternion (num_envs, 4) in wxyz format.
        v: Vector (num_envs, 3).

    Returns:
        Rotated vector (num_envs, 3).
    """
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    # t = 2 * cross(q_vec, v)
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    # result = v + q_w * t + cross(q_vec, t)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)
