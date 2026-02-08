# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def morphology_params(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env morphology parameter vector for heterogeneous robots.
    
    This observation term provides the policy with information about each
    environment's robot morphology, enabling morphology-adaptive behavior.
    
    The parameter vector is [N, 7]:
        [base_length, base_width, base_height, thigh_length, calf_length, 
         thigh_radius, parallel_abduction]
    
    Values are normalized to roughly [-1, 1] range for better learning.
    
    The morphology_params_tensor must be set on the environment by the
    ProceduralRobotEnv class during initialization.
    
    Returns:
        torch.Tensor: Shape [num_envs, 7] normalized morphology parameters.
    """
    if hasattr(env, "morphology_params_tensor"):
        return env.morphology_params_tensor
    else:
        # Fallback: return zeros if not available (e.g., homogeneous training)
        return torch.zeros(env.num_envs, 7, device=env.device)
