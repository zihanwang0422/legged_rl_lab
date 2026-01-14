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
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["target_height"] = 0.30
        
        # === 解决左右腿不对称和向右偏倒的问题 ===
        # 4. 左右腿关节对称奖励 - 鼓励左右腿关节位置对称,防止一侧迈步幅度过大
        self.rewards.joint_symmetry_l2.weight = -0.3
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
            ["FL_hip_joint", "RL_hip_joint"],  # 左前左后
            ["FR_hip_joint", "RR_hip_joint"],  # 右前右后
        ]
        # 5. 左右腿动作对称奖励 - 鼓励左右腿控制指令对称,防止控制不平衡
        self.rewards.action_symmetry_l2.weight = -0.2
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [
            ["FL_hip_joint", "RL_hip_joint"],  # 左侧腿
            ["FR_hip_joint", "RR_hip_joint"],  # 右侧腿
        ]
        # 6. 增强侧向速度跟踪奖励权重,帮助机器人更好地保持直线行走
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        # 7. 增加侧向角速度惩罚,防止左右倾斜
        self.rewards.ang_vel_xy_l2.weight = -0.1
        
        # === 解决trunk向左倾斜的问题 ===
        # 8. Roll角惩罚 - 强力惩罚机身左右倾斜(向左或向右)
        self.rewards.body_roll_l2.weight = -5.0
        # 9. 减弱flat_orientation权重,因为已经有专门的roll/pitch惩罚
        self.rewards.flat_orientation_l2.weight = -1.0
        
        # 10. 前腿 hip 关节角度偏差奖励(防止前腿外扩)
        self.rewards.front_hip_deviation_l1.weight = -0.5
        self.rewards.front_hip_deviation_l1.params["asset_cfg"].joint_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        #no base_lin_vel
        # self.observations.policy.base_lin_vel = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
    

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
