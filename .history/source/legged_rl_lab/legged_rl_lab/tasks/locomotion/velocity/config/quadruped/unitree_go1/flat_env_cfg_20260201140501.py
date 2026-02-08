# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from legged_rl_lab.assets.unitree import UNITREE_GO1_CFG  # isort: skip


@configclass
class UnitreeGo1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        # ====Terrain Cfg====
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # action
        self.actions.joint_pos.scale = 0.25

        # event - 参考官方配置
        self.events.randomize_push_robot = None
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 3.0)  # 官方值
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "trunk"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # ==================== Rewards Configuration ====================
        
        # ===== Rewards - 参考官方配置简化 =====
        # Tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        
        # Penalties - 只保留必要的
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -2.5
        
        # Joint & Action penalties
        self.rewards.joint_torques_l2.weight = -0.0002
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -1.0
        
        # Feet air time - 参考官方配置
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.5  # 官方值
        
        # Undesired contacts - 可选，先禁用看效果
        self.rewards.undesired_contacts = None
        
        # terminations
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "trunk"
        
        # commands - 扩大速度范围以支持高速运动
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)  
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0) 
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0) 


@configclass
class UnitreeGo1FlatEnvCfg_PLAY(UnitreeGo1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
  
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None