# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Reward functions for cross-embodied locomotion tasks.

This module is a self-contained copy of the project-level custom reward
functions (originally from velocity/mdp/rewards.py) with additional
cross-embodied specific rewards.  It has NO dependency on the
``legged_rl_lab.tasks.locomotion.velocity`` package.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Feet air-time
# ---------------------------------------------------------------------------

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds with single-stance constraint."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def double_flight_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize both feet being off the ground simultaneously (double-flight / bunny-hop)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    both_off = torch.sum(in_contact.int(), dim=1) == 0
    cmd_active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return (both_off & cmd_active).float()


# ---------------------------------------------------------------------------
# Feet slide
# ---------------------------------------------------------------------------

def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding on the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


# ---------------------------------------------------------------------------
# Foot clearance (with optional contact-sensor gate)
# ---------------------------------------------------------------------------

def foot_clearance_reward_humanoid(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    foot_scanner_cfgs: list[SceneEntityCfg] | None = None,
    contact_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """奖励摆动腿达到目标离地高度，同时不惩罚支撑腿。

    Args:
        foot_scanner_cfgs: 每只脚一个向下单射线 RayCaster sensor 配置列表（顺序须与
            asset_cfg.body_ids 一致）。若为 None，退化为用 env_origins Z 作为参考。
        contact_sensor_cfg: 脚部接触 sensor 配置（body_ids 须与 asset_cfg.body_ids 对应）。
            若提供，则在 velocity-based swing_mask 之上叠加「非接触」掩码，只有真正
            离地的脚才累计高度误差，防止侧向滑步等伪摆动行为游戏奖励。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    num_feet = foot_pos_w.shape[1]

    if foot_scanner_cfgs is not None:
        terrain_h = torch.stack(
            [env.scene.sensors[sc.name].data.ray_hits_w[:, 0, 2] for sc in foot_scanner_cfgs],
            dim=1,
        )
    else:
        terrain_h = env.scene.terrain.env_origins[:, 2].unsqueeze(1).expand(-1, num_feet)

    current_relative_h = foot_pos_w[:, :, 2] - terrain_h

    foot_vel_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    swing_mask = torch.tanh(tanh_mult * foot_vel_xy)

    # contact gate：只有真正离地的脚才参与 clearance 计算
    if contact_sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        is_in_contact = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0
        swing_mask = swing_mask * (~is_in_contact).float()

    h_error = torch.square(current_relative_h - target_height)
    return torch.exp(-torch.sum(h_error * swing_mask, dim=1) / std)


# ---------------------------------------------------------------------------
# Base height penalty
# ---------------------------------------------------------------------------

def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """单侧惩罚 base 高度低于 target_height（防止深蹲/劈叉导致躯干过低）。

    base 高于 target_height 时不惩罚；低于时给出二次惩罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_z = env.scene.terrain.env_origins[:, 2]
    h = asset.data.root_pos_w[:, 2] - env_z
    return torch.clamp(target_height - h, min=0.0).pow(2)


# ---------------------------------------------------------------------------
# Feet gait
# ---------------------------------------------------------------------------

def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward alternating foot contact pattern matching a desired gait clock."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


# ---------------------------------------------------------------------------
# Velocity tracking (yaw-frame)
# ---------------------------------------------------------------------------

def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward linear velocity tracking in the yaw-aligned body frame."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw angular velocity tracking in world frame."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


# ---------------------------------------------------------------------------
# Joint / action symmetry
# ---------------------------------------------------------------------------

def joint_mirror(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
    """Penalize joint position differences between mirrored joint pairs."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache:
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
    """Penalize action magnitude differences between mirrored joint pairs."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    action_clipped = torch.clamp(env.action_manager.action, -10.0, 10.0)
    for joint_pair in env.action_mirror_joints_cache:
        diff = torch.sum(
            torch.square(
                torch.abs(action_clipped[:, joint_pair[0][0]]) - torch.abs(action_clipped[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# ---------------------------------------------------------------------------
# Body orientation penalties
# ---------------------------------------------------------------------------

def body_roll_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize roll angle of the robot base."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 0])


def body_pitch_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize pitch angle of the robot base."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def joint_power(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize total joint mechanical power."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from default when command is nearly zero."""
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# ---------------------------------------------------------------------------
# Handstand / footstand helpers (kept for completeness)
# ---------------------------------------------------------------------------

def handstand_feet_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height_error = torch.sum(torch.square(feet_height - target_height), dim=1)
    return torch.exp(-feet_height_error / std**2)


def handstand_feet_on_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    return torch.all(first_air, dim=1).float()


def handstand_feet_air_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return torch.sum((last_air_time - threshold) * first_contact, dim=1)


def handstand_orientation_l2(
    env: ManagerBasedRLEnv,
    target_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    target_gravity_tensor = torch.tensor(target_gravity, device=env.device)
    return torch.sum(torch.square(asset.data.projected_gravity_b - target_gravity_tensor), dim=1)
