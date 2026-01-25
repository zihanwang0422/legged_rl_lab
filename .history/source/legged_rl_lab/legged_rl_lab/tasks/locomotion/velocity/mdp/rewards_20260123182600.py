# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def feet_height_body(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     asset_cfg: SceneEntityCfg,
#     target_height: float,
#     tanh_mult: float,
# ) -> torch.Tensor:
#     """Reward the swinging feet for clearing a specified height off the ground"""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
#     footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
#     cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
#         :, :
#     ].unsqueeze(1)
#     footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
#     for i in range(len(asset_cfg.body_ids)):
#         footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
#             asset.data.root_quat_w, cur_footpos_translated[:, i, :]
#         )
#         footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
#             asset.data.root_quat_w, cur_footvel_translated[:, i, :]
#         )
#     foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
#     foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
#     reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
#     reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward

def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """Penalize joint position differences between mirrored joints (e.g., left-right leg symmetry).
    
    This reward helps ensure symmetric gait patterns by penalizing differences between corresponding
    joints on opposite sides of the robot (e.g., left_hip vs right_hip).
    
    Args:
        env: The learning environment.
        asset_cfg: The asset configuration for the robot.
        mirror_joints: List of joint pairs to mirror, e.g., [[".*_hip_joint"], [".*R_hip_joint"]]
                       where each pair should have symmetric behavior.
    
    Returns:
        Penalty proportional to joint position differences between mirrored pairs.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    """Penalize action differences between mirrored joints (e.g., left-right leg symmetry).
    
    This reward helps ensure symmetric control commands by penalizing differences in action magnitudes
    between corresponding joints on opposite sides of the robot.
    
    Args:
        env: The learning environment.
        asset_cfg: The asset configuration for the robot.
        mirror_joints: List of joint pairs to mirror, e.g., [[".*L_hip_joint"], [".*R_hip_joint"]]
                       where each pair should receive symmetric actions.
    
    Returns:
        Penalty proportional to action differences between mirrored pairs.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward



def body_roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize roll angle (sideways tilt) of the robot base.
    
    This reward specifically targets the roll (x-axis rotation) of the robot body,
    which is useful for preventing sideways leaning/tilting during locomotion.
    
    Args:
        env: The learning environment.
        asset_cfg: The asset configuration for the robot.
    
    Returns:
        Penalty proportional to the square of the roll angle.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Roll is the first component of projected gravity in body frame (x-axis)
    # When upright, projected_gravity_b = [0, 0, -1]. Roll tilt causes x component to be non-zero
    return torch.square(asset.data.projected_gravity_b[:, 0])


def body_pitch_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize pitch angle (forward/backward tilt) of the robot base.
    
    This reward specifically targets the pitch (y-axis rotation) of the robot body,
    which is useful for preventing forward/backward leaning during locomotion.
    
    Args:
        env: The learning environment.
        asset_cfg: The asset configuration for the robot.
    
    Returns:
        Penalty proportional to the square of the pitch angle.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Pitch is the second component of projected gravity in body frame (y-axis)
    # When upright, projected_gravity_b = [0, 0, -1]. Pitch tilt causes y component to be non-zero
    return torch.square(asset.data.projected_gravity_b[:, 1])


# ===== Footstand/Handstand Rewards =====

def handstand_feet_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_height_error = torch.sum(torch.square(feet_height - target_height), dim=1)
    return torch.exp(-feet_height_error / std**2)


def handstand_feet_on_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    所有足部都在空中时奖励 1,否则奖励 0
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.all(first_air, dim=1).float()
    return reward


def handstand_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward continuous air time for specified feet, only when ALL feet are in the air.
    
    This ensures that the robot is only rewarded when all specified feet (e.g., front legs)
    are simultaneously airborne for the required duration.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    
    # Use minimum air time across all feet to ensure all are airborne
    min_air_time = torch.min(last_air_time, dim=1)[0]
    # Only reward when air time exceeds threshold and any foot just made contact
    reward = torch.clamp(min_air_time - threshold, min=0.0) * torch.any(first_contact, dim=1).float()
    
    return reward


def handstand_orientation_l2(
    env: ManagerBasedRLEnv, target_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Define the target gravity direction for an upright posture in the base frame
    target_gravity_tensor = torch.tensor(target_gravity, device=env.device)
    # Penalize deviation of the projected gravity vector from the target
    return torch.sum(torch.square(asset.data.projected_gravity_b - target_gravity_tensor), dim=1)


def reward_base_ang_vel_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """专门惩罚绕机身 X 轴的角速度（针对倒立画圆）"""
    asset: Articulation = env.scene[asset_cfg.name]
    # 只提取索引 0 (X轴)
    ang_vel_x = asset.data.root_com_ang_vel_b[:, 0]
    return torch.square(ang_vel_x)