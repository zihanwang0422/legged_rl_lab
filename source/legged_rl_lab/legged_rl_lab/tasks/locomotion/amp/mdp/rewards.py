# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Gait-periodic reward functions in the TienKung-Lab style.

Uses a smooth periodic clock signal to shape foot contact / motion patterns:
  * `gait_feet_frc_perio`        — penalise contact force during swing
  * `gait_feet_spd_perio`        — penalise foot speed during stance
  * `gait_feet_frc_support_perio`— reward strong contact during stance
  * `feet_clearance`             — reward target foot height during swing
  * `feet_y_distance`            — keep lateral spacing when not strafing
  * `feet_slide`                 — penalise sliding while in contact
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Gait phase clock
# ---------------------------------------------------------------------------

def _gait_phase(
    env: "ManagerBasedRLEnv",
    offset: float = 0.0,
    cycle: float = 0.7,
) -> torch.Tensor:
    """Normalised gait phase ∈ [0, 1) for a single foot."""
    step_dt = env.cfg.sim.dt * env.cfg.decimation
    t = env.episode_length_buf.float() * step_dt / cycle
    return (t + offset) % 1.0


def _gait_clock(
    phase: torch.Tensor,
    air_ratio: float,
    delta_t: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (I_frc, I_spd) where:

    * ``I_frc`` ≈ 1 during swing, 0 during stance (use to gate swing rewards)
    * ``I_spd`` ≈ 1 during stance, 0 during swing (use to gate stance rewards)

    Linear interpolation is used at swing↔stance boundaries (width ``2·delta_t``)
    so the gradient is smooth.  Exactly mirrors TienKung-Lab's formulation.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stance_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1.0 - delta_t))

    trans1 = phase < delta_t                          # cycle wrap-in (entering swing)
    trans2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))  # swing→stance
    trans3 = phase > (1.0 - delta_t)                  # stance→swing (cycle wrap-out)

    I_frc = (
        1.0 * swing_flag.float()
        + (0.5 + phase / (2.0 * delta_t)) * trans1.float()
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans2.float()
        + 0.0 * stance_flag.float()
        + (phase - 1.0 + delta_t) / (2.0 * delta_t) * trans3.float()
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


# ---------------------------------------------------------------------------
# Periodic foot rewards (TienKung-Lab style, exp shaping)
# ---------------------------------------------------------------------------

def gait_feet_frc_perio(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    cycle: float = 0.7,
    offset_l: float = 0.0,
    offset_r: float = 0.5,
    air_ratio: float = 0.4,
    force_sigma: float = 200.0,
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Penalise contact force during swing phase.

    For each foot: ``I_frc * clip(1 - f/force_sigma, 0, 1)`` — linear ramp from
    1 (no force) down to 0 (force ≥ force_sigma N).  Linear shaping gives a
    constant gradient over the full operating range; the previous ``exp(-f²/σ)``
    form saturates to zero whenever forces are far from optimum, so a policy
    that hasn't yet learned to lift its feet sees no signal at all.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3], dim=-1)
    # forces: (num_envs, 2)

    phase_l = _gait_phase(env, offset=offset_l, cycle=cycle)
    phase_r = _gait_phase(env, offset=offset_r, cycle=cycle)
    I_frc_l, _ = _gait_clock(phase_l, air_ratio, delta_t)
    I_frc_r, _ = _gait_clock(phase_r, air_ratio, delta_t)

    shaped_l = torch.clamp(1.0 - forces[:, 0] / force_sigma, min=0.0, max=1.0)
    shaped_r = torch.clamp(1.0 - forces[:, 1] / force_sigma, min=0.0, max=1.0)
    return 0.5 * (I_frc_l * shaped_l + I_frc_r * shaped_r)


def gait_feet_spd_perio(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cycle: float = 0.7,
    offset_l: float = 0.0,
    offset_r: float = 0.5,
    air_ratio: float = 0.4,
    speed_sigma: float = 0.5,
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Penalise foot speed during stance phase.

    For each foot: ``I_spd * clip(1 - v/speed_sigma, 0, 1)``.  Linear ramp so
    the gradient is constant — encourages a planted foot during stance without
    saturating away from optimum.
    """
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
    foot_speeds = torch.norm(asset.data.body_lin_vel_w[:, body_ids, :], dim=-1)

    phase_l = _gait_phase(env, offset=offset_l, cycle=cycle)
    phase_r = _gait_phase(env, offset=offset_r, cycle=cycle)
    _, I_spd_l = _gait_clock(phase_l, air_ratio, delta_t)
    _, I_spd_r = _gait_clock(phase_r, air_ratio, delta_t)

    shaped_l = torch.clamp(1.0 - foot_speeds[:, 0] / speed_sigma, min=0.0, max=1.0)
    shaped_r = torch.clamp(1.0 - foot_speeds[:, 1] / speed_sigma, min=0.0, max=1.0)
    return 0.5 * (I_spd_l * shaped_l + I_spd_r * shaped_r)


def gait_feet_frc_support_perio(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    cycle: float = 0.7,
    offset_l: float = 0.0,
    offset_r: float = 0.5,
    air_ratio: float = 0.4,
    force_sigma: float = 1.0e4,
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Reward firm support force during stance phase.

    For each foot: ``I_spd * clip(f / force_sigma, 0, 1)``.  Linear ramp from
    0 (no contact) to 1 (force ≥ ``force_sigma`` N).  Encourages the foot to
    actually press into the ground when it should be supporting; gradient is
    constant rather than dying as ``f → 0`` like the previous exp form.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3], dim=-1)

    phase_l = _gait_phase(env, offset=offset_l, cycle=cycle)
    phase_r = _gait_phase(env, offset=offset_r, cycle=cycle)
    _, I_spd_l = _gait_clock(phase_l, air_ratio, delta_t)
    _, I_spd_r = _gait_clock(phase_r, air_ratio, delta_t)

    shaped_l = torch.clamp(forces[:, 0] / force_sigma, min=0.0, max=1.0)
    shaped_r = torch.clamp(forces[:, 1] / force_sigma, min=0.0, max=1.0)
    return 0.5 * (I_spd_l * shaped_l + I_spd_r * shaped_r)


# ---------------------------------------------------------------------------
# Foot-clearance / spacing / slide
# ---------------------------------------------------------------------------

def feet_clearance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cycle: float = 0.7,
    offset_l: float = 0.0,
    offset_r: float = 0.5,
    air_ratio: float = 0.4,
    target_height: float = 0.12,
    height_sigma: float = 0.025,
    delta_t: float = 0.02,
) -> torch.Tensor:
    """Reward target foot-clearance during swing.

    Reward per foot: ``I_frc * clip(z / target_height, 0, 1)``.  Linear ramp
    that hits 1 at the target height — gives a constant gradient even when the
    foot is barely off the ground (the previous gaussian shape was ~0 below
    ~5 cm, so a shuffling policy saw essentially no signal).  ``height_sigma``
    is kept in the signature for backwards-compatibility but is no longer used.
    """
    del height_sigma  # unused; kept for backward compat
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
    foot_z = asset.data.body_pos_w[:, body_ids, 2]  # (num_envs, 2)

    phase_l = _gait_phase(env, offset=offset_l, cycle=cycle)
    phase_r = _gait_phase(env, offset=offset_r, cycle=cycle)
    I_frc_l, _ = _gait_clock(phase_l, air_ratio, delta_t)
    I_frc_r, _ = _gait_clock(phase_r, air_ratio, delta_t)

    shaped_l = torch.clamp(foot_z[:, 0] / target_height, min=0.0, max=1.0)
    shaped_r = torch.clamp(foot_z[:, 1] / target_height, min=0.0, max=1.0)
    return 0.5 * (I_frc_l * shaped_l + I_frc_r * shaped_r)


def feet_y_distance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_distance: float = 0.20,
    command_name: str = "base_velocity",
    y_vel_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalise lateral foot-spacing deviation (only when not strafing).

    Computes |y_left_b − y_right_b − target_distance| in the base frame.
    Disabled when |cmd_v_y| ≥ ``y_vel_threshold`` so a side-step command can
    legitimately widen the stance.
    """
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
    assert len(body_ids) == 2, "feet_y_distance expects exactly 2 foot bodies (L, R)"

    root_pos = asset.data.root_link_pos_w
    root_quat = asset.data.root_link_quat_w
    foot_l_w = asset.data.body_pos_w[:, body_ids[0], :] - root_pos
    foot_r_w = asset.data.body_pos_w[:, body_ids[1], :] - root_pos
    foot_l_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat), foot_l_w)
    foot_r_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat), foot_r_w)

    err = torch.abs(foot_l_b[:, 1] - foot_r_b[:, 1] - target_distance)

    cmd = env.command_manager.get_command(command_name)
    not_strafing = (torch.abs(cmd[:, 1]) < y_vel_threshold).float()
    return err * not_strafing


def feet_slide(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty for foot horizontal speed while in contact.

    ``Σ_feet  ‖v_xy‖ * 1[is_in_contact]``.  This complements the periodic
    spd reward by directly killing any horizontal velocity the moment a foot
    is detected to be on the ground (regardless of phase).
    """
    sensor = env.scene[sensor_cfg.name]
    in_contact = (
        torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3], dim=-1)
        > contact_force_threshold
    )

    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
    foot_xy_speed = torch.norm(asset.data.body_lin_vel_w[:, body_ids, :2], dim=-1)

    return torch.sum(foot_xy_speed * in_contact.float(), dim=1)
