# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-specific observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_body_pos_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Selected body positions in the anchor body frame."""
    asset = env.scene[anchor_cfg.name]
    anchor_ids, _ = asset.find_bodies(anchor_cfg.body_names, preserve_order=True)
    body_ids, _ = asset.find_bodies(body_cfg.body_names, preserve_order=True)

    anchor_pos_w = asset.data.body_pos_w[:, anchor_ids[0], :]
    anchor_quat_w = asset.data.body_quat_w[:, anchor_ids[0], :]
    body_pos_w = asset.data.body_pos_w[:, body_ids, :]

    rel_w = body_pos_w - anchor_pos_w.unsqueeze(1)
    num_bodies = body_pos_w.shape[1]
    anchor_quat = anchor_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
    pos_b = quat_apply_inverse(
        anchor_quat.reshape(-1, 4),
        rel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Selected body orientations relative to the anchor body as 6D rotations."""
    asset = env.scene[anchor_cfg.name]
    anchor_ids, _ = asset.find_bodies(anchor_cfg.body_names, preserve_order=True)
    body_ids, _ = asset.find_bodies(body_cfg.body_names, preserve_order=True)

    anchor_quat_w = asset.data.body_quat_w[:, anchor_ids[0], :]
    body_quat_w = asset.data.body_quat_w[:, body_ids, :]
    num_bodies = body_quat_w.shape[1]

    anchor_quat = anchor_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
    rel_quat = math_utils.quat_mul(
        math_utils.quat_conjugate(anchor_quat.reshape(-1, 4)),
        body_quat_w.reshape(-1, 4),
    )
    mat = math_utils.matrix_from_quat(rel_quat).reshape(env.num_envs, num_bodies, 3, 3)
    return mat[..., :2].reshape(env.num_envs, -1)


def robot_body_lin_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Selected body linear velocities in each body's local frame."""
    del anchor_cfg
    asset = env.scene[body_cfg.name]
    body_ids, _ = asset.find_bodies(body_cfg.body_names, preserve_order=True)

    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_ids, :]
    body_quat_w = asset.data.body_quat_w[:, body_ids, :]
    num_bodies = body_lin_vel_w.shape[1]
    lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    return lin_vel_b.reshape(env.num_envs, -1)


def robot_body_ang_vel_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """Selected body angular velocities in each body's local frame."""
    del anchor_cfg
    asset = env.scene[body_cfg.name]
    body_ids, _ = asset.find_bodies(body_cfg.body_names, preserve_order=True)

    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_ids, :]
    body_quat_w = asset.data.body_quat_w[:, body_ids, :]
    num_bodies = body_ang_vel_w.shape[1]
    ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    return ang_vel_b.reshape(env.num_envs, -1)
