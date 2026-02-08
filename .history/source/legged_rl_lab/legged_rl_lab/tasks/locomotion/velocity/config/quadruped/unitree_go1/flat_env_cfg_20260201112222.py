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

        # event
        self.events.randomize_push_robot = None
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.8, 1.2)  # 合理的质量变化
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
        
        # ===== General Rewards =====
        self.rewards.is_alive = None
        self.rewards.is_terminated = None
        
        # ===== Base Rewards =====
        # Tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        
        # Base penalties
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["target_height"] = 0.34  # Go1正常站立高度
        self.rewards.body_lin_acc_l2 = None
        self.rewards.base_ang_vel_x_l2 = None
        
        # ===== Joint Rewards =====
        self.rewards.joint_torques_l2.weight = -0.0002
        self.rewards.joint_vel_l1 = None
        self.rewards.joint_vel_l2 = None
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_deviation_l1 = None
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_vel_limits = None
        self.rewards.applied_torque_limits = None
        self.rewards.joint_power = None
        
        # Action penalties
        self.rewards.action_rate_l2 = None
        self.rewards.action_l2 = None
        
        # ===== Contact Rewards =====
        # Undesired contacts (防止小腿和大腿贴地)
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(calf|thigh)"
        self.rewards.desired_contacts = None
        self.rewards.contact_forces = None
        
        # Feet rewards
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_height = None
        self.rewards.feet_slide = None
        
        # ===== Other/Custom Rewards =====
        # Standing still
        self.rewards.stand_till.weight = -2.0
        self.rewards.stand_till.params["command_name"] = "base_velocity"
        
        # Body orientation
        self.rewards.body_roll_l2.weight = -5.0
        self.rewards.body_pitch_l2 = None
        
        # Diagonal gait symmetry (Trot Gait: FL+RR, FR+RL)
        self.rewards.joint_symmetry_l2.weight = -0.1
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
            ["FL_.*_joint", "RR_.*_joint"],  # 前左 + 后右 (对角线1)
            ["FR_.*_joint", "RL_.*_joint"],  # 前右 + 后左 (对角线2)
        ]
        self.rewards.action_symmetry_l2.weight = -0.05
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [
            ["FL_.*_joint", "RR_.*_joint"],  # 前左 + 后右 (对角线1)
            ["FR_.*_joint", "RL_.*_joint"],  # 前右 + 后左 (对角线2)
        ]
        
        # ===== Handstand/Footstand Rewards =====
        self.rewards.handstand_feet_height_exp = None
        self.rewards.handstand_feet_on_air = None
        self.rewards.handstand_feet_air_time = None
        self.rewards.handstand_orientation_l2 = None
        
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