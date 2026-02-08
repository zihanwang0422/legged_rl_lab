# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from legged_rl_lab.assets.unitree import UNITREE_GO1_CFG  # isort: skip
import isaaclab.terrains as terrain_gen

@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        
        
        #------------------------------- Terrain -------------------------------
        # 减少地形网格数量以节省内存
        self.scene.terrain.terrain_generator.num_rows = 3  # 默认 10
        self.scene.terrain.terrain_generator.num_cols = 3  # 默认 10
        self.scene.terrain.terrain_generator.horizontal_scale = 0.2  # 增大以减少顶点数
        self.scene.terrain.terrain_generator.vertical_scale = 0.005
        self.scene.terrain.terrain_generator.sub_terrains = {
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.25,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "inverted_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.25,
                step_height_range=(0.05, 0.23),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.25,
                slope_range=(0.0, 0.4),
                platform_width=3.0,
                border_width=0.25,
                inverted=False,
            ),
            "inverted_slopes": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.25,
                slope_range=(0.0, 0.4),
                platform_width=3.0,
                border_width=0.25,
            ),
        }

        # action
        self.actions.joint_pos.scale = 0.25

        #------------------------------- Event -------------------------------
        self.events.randomize_push_robot = None
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 3.0)
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
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        
        # Base penalties
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.base_height_l2 = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.base_ang_vel_x_l2 = None
        
        # ===== Joint Rewards =====
        self.rewards.joint_torques_l2.weight = -0.0002
        self.rewards.joint_vel_l1 = None
        self.rewards.joint_vel_l2 = None
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_deviation_l1 = None
        self.rewards.joint_pos_limits = None
        self.rewards.joint_vel_limits = None
        self.rewards.applied_torque_limits = None
        self.rewards.joint_power = None
        
        # Action penalties
        self.rewards.action_rate_l2 = None
        self.rewards.action_l2 = None
        
        # ===== Contact Rewards =====
        self.rewards.undesired_contacts = None
        self.rewards.desired_contacts = None
        self.rewards.contact_forces = None
        
        # Feet rewards
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_height = None
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_foot"
    
        # ===== Other/Custom Rewards =====
        # Standing still
        self.rewards.stand_till.weight = -2.0
        self.rewards.stand_till.params["command_name"] = "base_velocity"
        
        # Body orientation penalties
        self.rewards.body_roll_l2 = None
        self.rewards.body_pitch_l2 = None
        
        # Symmetry rewards
        self.rewards.joint_symmetry_l2 = None
        self.rewards.action_symmetry_l2 = None
        
        # ==================== Handstand/Footstand Rewards ====================
        self.rewards.handstand_feet_height_exp = None
        self.rewards.handstand_feet_on_air = None
        self.rewards.handstand_feet_air_time = None
        self.rewards.handstand_orientation_l2 = None
        
        #------------------------------- Terminations -------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "trunk"
        
        #------------------------------- Commands -------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)  
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8) 
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0) 


@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
