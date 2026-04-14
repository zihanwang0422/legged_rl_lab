# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for terrain-aware operations (cross-embodied tasks)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_terrain_column_range(terrain_cfg, terrain_name: str, device) -> tuple[int, int] | None:
    """Helper function to calculate column range for a terrain type."""
    if terrain_cfg.sub_terrains is None or terrain_name not in terrain_cfg.sub_terrains:
        return None

    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())
    proportions = torch.tensor([sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()], device=device)
    proportions = proportions / proportions.sum()
    cumsum_props = torch.cumsum(proportions, dim=0)

    terrain_idx = sub_terrain_names.index(terrain_name)
    col_start = round((0.0 if terrain_idx == 0 else cumsum_props[terrain_idx - 1].item()) * terrain_cfg.num_cols)
    col_end = round(cumsum_props[terrain_idx].item() * terrain_cfg.num_cols)

    return (col_start, col_end)


def is_env_assigned_to_terrain(env: ManagerBasedEnv, terrain_name: str) -> torch.Tensor:
    """Check which environments are initially assigned to the specified terrain type."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range
    return (terrain.terrain_types >= col_start) & (terrain.terrain_types < col_end)


def is_robot_on_terrain(env: ManagerBasedEnv, terrain_name: str, asset_name: str = "robot") -> torch.Tensor:
    """Check which robots are currently standing on the specified terrain type."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range

    asset = env.scene[asset_name]
    robot_pos_w = asset.data.root_pos_w[:, :2]

    terrain_origins = terrain.terrain_origins
    num_rows, num_cols, _ = terrain_origins.shape
    terrain_origins_2d = terrain_origins[:, :, :2].reshape(num_rows * num_cols, 2)

    distances = torch.cdist(robot_pos_w, terrain_origins_2d)
    closest_flat_idx = torch.argmin(distances, dim=1)
    col_idx = closest_flat_idx % num_cols

    return (col_idx >= col_start) & (col_idx < col_end)
