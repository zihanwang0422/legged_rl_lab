# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-specific event functions — Reference State Initialization (RSI)."""

from __future__ import annotations

import os
import warnings

import torch

from isaaclab.managers import SceneEntityCfg

from legged_rl_lab.managers import MotionLoader


def reset_from_reference_motion(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot state by sampling a random reference motion frame (RSI).

    Implements Reference State Initialization for AMP training:
    samples a uniformly random frame from the loaded reference motion,
    then sets the robot's root pose, root velocity, and joint state to
    match that frame.

    The MotionLoader is initialized lazily on the first call (cached on
    ``env._amp_rsi_loader``).  It reads ``env.cfg.amp_motion_files`` and
    ``env.cfg.robot_type`` from the environment config — both are set in
    the task-specific environment config (e.g. ``UnitreeG1AMPFlatEnvCfg``).

    If motion data is unavailable or the loaded data has no raw state
    (e.g. pre-processed ``.npy`` files), a warning is emitted once and
    the function becomes a no-op.  Set a fallback event (``reset_base`` +
    ``reset_robot_joints``) in the base config when needed.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments to reset.
        asset_cfg: Scene entity config for the robot articulation.
    """
    # ---------------------------------------------------------------
    # Lazy MotionLoader initialization — cached on env instance
    # ---------------------------------------------------------------
    if not hasattr(env, "_amp_rsi_loader"):
        loader: MotionLoader | None = None
        motion_path: str = getattr(env.cfg, "amp_motion_files", "")
        robot_type: str = getattr(env.cfg, "robot_type", "g1")

        if motion_path and os.path.exists(motion_path):
            try:
                loader = MotionLoader(device=env.device, robot=robot_type)
                loader.load(motion_path)
                if not loader.has_state_data:
                    warnings.warn(
                        "[RSI] MotionLoader loaded data but found no raw state tensors "
                        "(pre-processed .npy/.pt files do not carry state). "
                        "reset_from_reference_motion will be a no-op. "
                        "Use .npz or .csv source files for RSI.",
                        stacklevel=1,
                    )
                    loader = None
            except Exception as exc:
                warnings.warn(
                    f"[RSI] Failed to load motion data from '{motion_path}': {exc}. "
                    "reset_from_reference_motion will be a no-op.",
                    stacklevel=1,
                )
                loader = None
        else:
            warnings.warn(
                f"[RSI] Motion file path not found or not set: '{motion_path}'. "
                "reset_from_reference_motion will be a no-op.",
                stacklevel=1,
            )

        env._amp_rsi_loader = loader  # type: ignore[attr-defined]

    loader: MotionLoader | None = env._amp_rsi_loader  # type: ignore[attr-defined]
    if loader is None:
        return

    n = len(env_ids)
    if n == 0:
        return

    states = loader.get_random_state(n)
    asset = env.scene[asset_cfg.name]

    # Root position:
    #   - x, y: from env origin so each env stays in its own patch
    #   - z: reference motion height + env origin z (handles terrain offsets)
    env_origins = env.scene.env_origins[env_ids]  # (n, 3)
    root_pos = env_origins.clone()
    root_pos[:, 2] = env_origins[:, 2] + states["root_pos"][:, 2]

    # Root orientation: directly from reference motion [w, x, y, z]
    root_quat = states["root_quat"]  # (n, 4)

    # Root pose: (n, 7) = [pos(3), quat_wxyz(4)]
    root_pose = torch.cat([root_pos, root_quat], dim=-1)

    # Root velocity: world frame (n, 6) = [lin_vel(3), ang_vel(3)]
    root_vel = torch.cat([states["root_lin_vel_w"], states["root_ang_vel_w"]], dim=-1)

    # Joint state (absolute positions, BFS order)
    joint_pos = states["joint_pos"]  # (n, num_dof)
    joint_vel = states["joint_vel"]  # (n, num_dof)

    # Write to simulation
    asset.write_root_pose_to_sim(root_pose, env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
