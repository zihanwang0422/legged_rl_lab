# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Procedural / morphology helpers for cross-embodied locomotion tasks.

Public API
----------
Observation terms (use in ObsTerm):
    phase(env, cycle_time)          → (N, 2)
    morphology_params(env)          → (N, morph_dim)

Env-init helpers (call once from __init__ after super()):
    modify_procedural_articulations(env)   – adjust joint limits via builder
    setup_morphology_params(env)           – generic: QuadrupedBuilder / BipedBuilder
    setup_cross_embodied_morphology_params(env) – G1+Go2 specific (3-dim)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    """Two-dim sinusoidal gait-phase encoding ``[sin, cos]``.

    Shape: ``(num_envs, 2)``.
    """
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    t = env.episode_length_buf[:, None] * env.step_dt / cycle_time  # (N, 1)
    return torch.cat([torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=-1)


def morphology_params(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the per-env morphology parameter vector.

    Reads ``env.morphology_params_tensor`` set by one of the setup helpers
    below.  Falls back to zeros when not available.

    Shape: ``(num_envs, morph_dim)``.
    """
    if hasattr(env, "morphology_params_tensor"):
        return env.morphology_params_tensor
    morph_dim = getattr(env, "morphology_params_dim", 7)
    return torch.zeros(env.num_envs, morph_dim, device=env.device)


# ---------------------------------------------------------------------------
# Env-init helpers
# ---------------------------------------------------------------------------


def modify_procedural_articulations(env: ManagerBasedRLEnv) -> None:
    """Adjust joint limits / default positions via metamorphosis builders.

    Mirrors ``ProceduralRobotEnv._modify_procedural_articulations``.
    Call once from ``__init__`` after ``super().__init__()``.
    """
    if "robot" not in env.scene.articulations:
        return

    robot = env.scene.articulations["robot"]

    for builder_name in ("QuadrupedBuilder", "QuadWheelBuilder", "BipedBuilder"):
        try:
            from metamorphosis import builder as _b
            BuilderCls = getattr(_b, builder_name)
            try:
                instance = BuilderCls.get_instance()
                if instance is not None and len(instance.params) > 0:
                    if hasattr(BuilderCls, "modify_articulation"):
                        BuilderCls.modify_articulation(robot)
                        print(f"[INFO] Modified articulation via {builder_name} "
                              f"({len(instance.params)} robots)")
            except (RuntimeError, KeyError):
                pass
        except (ImportError, AttributeError):
            pass


def setup_morphology_params(env: ManagerBasedRLEnv) -> None:
    """Set ``env.morphology_params_tensor`` from metamorphosis builder params.

    Tries QuadrupedBuilder first, then BipedBuilder.
    Falls back to zeros (dim=7) when no builder is available.
    """
    # --- QuadrupedBuilder ---------------------------------------------------
    try:
        from metamorphosis.builder import QuadrupedBuilder
        try:
            builder = QuadrupedBuilder.get_instance()
            if builder is not None and len(builder.params) > 0:
                params_list = [
                    [
                        p.base_length, p.base_width, p.base_height,
                        p.thigh_length, p.calf_length, p.thigh_radius,
                        float(p.parallel_abduction),
                    ]
                    for p in builder.params
                ]
                raw = torch.tensor(params_list, device=env.device, dtype=torch.float32)
                mean = torch.tensor([[0.75, 0.35, 0.20, 0.50, 0.50, 0.04, 0.5]], device=env.device)
                std  = torch.tensor([[0.25, 0.05, 0.05, 0.30, 0.30, 0.01, 0.5]], device=env.device)
                env.morphology_params_tensor = (raw - mean) / std
                env.morphology_params_dim = env.morphology_params_tensor.shape[-1]
                print(f"[INFO] Morphology params set via QuadrupedBuilder "
                      f"(shape {env.morphology_params_tensor.shape})")
                return
        except (RuntimeError, KeyError):
            pass
    except ImportError:
        pass

    # --- BipedBuilder -------------------------------------------------------
    try:
        from metamorphosis.builder import BipedBuilder
        try:
            builder = BipedBuilder.get_instance()
            if builder is not None and len(builder.params) > 0:
                params_list = [
                    [
                        p.torso_link_length, p.torso_link_width, p.torso_link_height,
                        p.pelvis_height, p.hip_spacing,
                        p.hip_pitch_link_length, p.hip_roll_link_length, p.hip_yaw_link_length,
                        p.knee_link_length, p.ankle_roll_link_length,
                    ]
                    for p in builder.params
                ]
                raw  = torch.tensor(params_list, device=env.device, dtype=torch.float32)
                mean = torch.tensor(
                    [[0.13, 0.22, 0.11, 0.065, 0.20, 0.045, 0.045, 0.30, 0.31, 0.215]],
                    device=env.device,
                )
                std  = torch.tensor(
                    [[0.03, 0.04, 0.03, 0.015, 0.04, 0.015, 0.015, 0.08, 0.09, 0.035]],
                    device=env.device,
                )
                env.morphology_params_tensor = (raw - mean) / std
                env.morphology_params_dim = env.morphology_params_tensor.shape[-1]
                print(f"[INFO] Morphology params set via BipedBuilder "
                      f"(shape {env.morphology_params_tensor.shape})")
                return
        except (RuntimeError, KeyError):
            pass
    except ImportError:
        pass

    # --- Fallback -----------------------------------------------------------
    env.morphology_params_dim = 7
    env.morphology_params_tensor = torch.zeros(env.num_envs, 7, device=env.device)


def setup_cross_embodied_morphology_params(env: ManagerBasedRLEnv) -> None:
    """Set ``env.morphology_params_tensor`` for the fixed G1+Go2 cross-embodied env.

    Produces a lightweight 3-dim vector per environment:
      [robot_type, dof_norm, height_norm]

    - robot_type : G1 = +1, Go2 = -1
    - dof_norm   : normalised over [12, 29]  →  Go2 = -1, G1 = +1
    - height_norm: normalised over [0.38, 0.78] → Go2 ≈ -1, G1 ≈ +1
    """
    is_g1 = env.is_g1_env.float()

    dof_raw    = torch.where(env.is_g1_env, torch.full_like(is_g1, 29.0), torch.full_like(is_g1, 12.0))
    dof_norm   = (dof_raw - 20.5) / 8.5

    height_raw  = torch.where(env.is_g1_env, torch.full_like(is_g1, 0.78), torch.full_like(is_g1, 0.38))
    height_norm = (height_raw - 0.58) / 0.20

    env.morphology_params_tensor = torch.stack(
        [is_g1 * 2.0 - 1.0, dof_norm, height_norm], dim=1
    )
    env.morphology_params_dim = 3


# ---------------------------------------------------------------------------
# Procedural robot environment
# ---------------------------------------------------------------------------


class ProceduralRobotEnv(ManagerBasedRLEnv):
    """Custom ManagerBasedRLEnv for procedurally generated robots.

    After ``super().__init__()`` the environment:
    1. Calls :func:`modify_procedural_articulations` to adjust joint limits
       and default positions via the metamorphosis builder.
    2. Calls :func:`setup_morphology_params` to populate
       ``env.morphology_params_tensor`` for the morphology observation.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        modify_procedural_articulations(self)
        setup_morphology_params(self)
