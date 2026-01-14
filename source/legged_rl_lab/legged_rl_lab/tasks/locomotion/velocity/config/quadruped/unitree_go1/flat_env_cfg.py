# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo1RoughEnvCfg


@configclass
class UnitreeGo1FlatEnvCfg(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        
        # === 解决膝关节角度过小和小腿贴地问题的奖励 ===
        # 1. 惩罚小腿(calf)和大腿(thigh)接触地面 - 防止小腿贴地
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(calf|thigh)"
        # 2. 惩罚关节超出软限制 - 防止膝关节角度过小(腿伸太直)
        self.rewards.dof_pos_limits.weight = -1.0
        # 3. 维持机身高度 - 鼓励机器人保持适当的站立高度
        self.rewards.base_height_l2.weight = -2.0 #
        self.rewards.base_height_l2.params["target_height"] = 0.30
        
        # ====Make trunk flat====
        # 6. 增强侧向速度跟踪奖励权重,帮助机器人更好地保持直线行走
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        # 7. 增加侧向角速度惩罚,防止左右倾斜
        self.rewards.ang_vel_xy_l2.weight = -0.1
        
        self.rewards.body_roll_l2.weight = -5.0 #
        self.rewards.flat_orientation_l2.weight = -1.0 #
        
        # ====Terrain Cfg====
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        # ===Symmetric rewards for stable gait===
        self.rewards.joint_symmetry_l2.weight = -0.3
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
            ["FL_hip_joint", "RL_hip_joint"],   
            ["FR_hip_joint", "RR_hip_joint"],  
        ]

        self.rewards.action_symmetry_l2.weight = -0.2
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [
            ["FL_hip_joint", "RL_hip_joint"],  
            ["FR_hip_joint", "RR_hip_joint"],  
        ]
        
        # commands - 扩大速度范围以支持高速运动
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)  # 扩展到1.0m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)  # 相应扩展侧向速度
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)  # 扩展角速度

        
    

class UnitreeGo1FlatEnvCfg_PLAY(UnitreeGo1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
