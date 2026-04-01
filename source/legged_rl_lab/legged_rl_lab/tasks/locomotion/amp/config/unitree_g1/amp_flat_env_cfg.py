# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP environment configurations for Unitree G1 humanoid robot."""

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_rl_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAMPRoughEnvCfg
from legged_rl_lab import LEGGED_RL_LAB_ROOT_DIR
import legged_rl_lab.tasks.locomotion.amp.mdp as mdp

##
# Pre-defined configs
##
from legged_rl_lab.assets.unitree import UNITREE_G1_29DOF_CFG  # isort: skip

@configclass
class UnitreeG1AMPFlatEnvCfg(LocomotionAMPRoughEnvCfg):
    """Unitree G1 humanoid AMP environment on flat terrain."""

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ----------------------------- Scene -----------------------------
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # No height scanner
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        
        # No terrain curriculum
        self.curriculum.terrain_levels = None

        # ----------------------------- Observations -----------------------------
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # AMP: 足部位置特征已禁用，保持 AMASS NPZ 和 LAFAN1 CSV 两种数据源
        # 的 obs 维度一致（均为 64d = 29+29+3+3）。
        # 若需要足部位置，需同时在 motion_loader G1 profile 中启用
        # foot_body_indices 并确保所有数据源都包含刚体位置信息。
        self.observations.amp.foot_positions = None

        # ----------------------------- Actions -----------------------------
        self.actions.joint_pos.scale = 0.25

        # ----------------------------- Events -----------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.push_robot = None

        # Replace uniform reset with Reference State Initialization (RSI):
        # samples a random reference motion frame and sets the robot state to it.
        # This eliminates the AMP dead-lock where the discriminator trivially
        # identifies policy (always standing) vs expert (walking).
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.events.reset_from_ref = EventTerm(
            func=mdp.reset_from_reference_motion,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ----------------------------- Rewards -----------------------------
        # 生存奖励：减少 illegal contact（原 0, 导致 23% 倒地）
        # 轻量权重 0.5，避免压制风格奖励信号
        self.rewards.is_alive.weight = 0.5

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # 平地姿态稳定惩罚：-1.0 过重，限制了腿部自然摆动（走路需要轻微俯仰）
        # 降至 -0.3，仍可防止大角度倾倒但不压制步态
        self.rewards.flat_orientation_l2.weight = -0.3

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.joint_acc_l2.weight = -1.25e-7

        self.rewards.action_rate_l2.weight = -0.001

        # Contact rewards
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=self.foot_link_name
        )

        # ----------------------------- Terminations -----------------------------
        self.terminations.illegal_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[self.base_link_name]
        )

        # ----------------------------- Commands -----------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ----------------------------- AMP Motion Data -----------------------------
        self.robot_type = "g1"
        self.amp_motion_files = os.path.join(
            LEGGED_RL_LAB_ROOT_DIR, "data", "motion", "AMASS_Retargeted_for_G1", "g1"
        )

@configclass
class UnitreeG1AMPFlatEnvCfg_PLAY(UnitreeG1AMPFlatEnvCfg):
    """Unitree G1 AMP environment for visualisation / play."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.lin_vel_cmd_levels = None
