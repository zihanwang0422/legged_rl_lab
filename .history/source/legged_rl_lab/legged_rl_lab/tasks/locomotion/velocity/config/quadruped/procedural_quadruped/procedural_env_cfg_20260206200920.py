# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0
"""
Procedural Quadruped Environment Configuration.

This module provides environment configurations for training RL agents on
procedurally generated quadruped robots using the metamorphosis framework.
Each environment instance can have a unique robot morphology, enabling
morphology-agnostic policy learning.
"""

from __future__ import annotations

import numpy as np

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Import metamorphosis procedural generation
from metamorphosis.asset_cfg import ProceduralQuadrupedCfg
from metamorphosis.builder import QuadrupedBuilder


##
# Procedural Quadruped ArticulationCfg
##

PROCEDURAL_QUADRUPED_CFG = ArticulationCfg(
    spawn=ProceduralQuadrupedCfg(
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        # Morphology parameter ranges - can be customized
        base_length_range=(0.5, 1.0),
        base_width_range=(0.3, 0.4),
        base_height_range=(0.1, 0.2),
        leg_length_range=(0.5, 0.9),
        calf_length_ratio=(0.85, 1.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            ".*_hip_joint": 0.0,
            "F[L,R]_thigh_joint": np.pi / 4,
            "R[L,R]_thigh_joint": np.pi / 4,
            ".*_calf_joint": -np.pi / 2,
        },
        pos=(0, 0, 0.6),  # Higher initial position to accommodate varying leg lengths
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            stiffness=25.0,
            damping=1.0,
            armature=0.01,
            friction=0.1,
        ),
    },
)


@configclass
class ProceduralQuadrupedFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Environment configuration for procedural quadruped on flat terrain.
    
    This configuration enables training with morphologically diverse quadrupeds.
    Each environment instance spawns a unique robot with different body proportions.
    
    Key features:
    - Procedural robot generation via metamorphosis
    - Flat terrain for stable initial training
    - Adaptive observation space (normalized joint positions)
    - Domain randomization for robust policies
    """
    
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # ==================== Scene Configuration ====================
        # Use procedural quadruped
        self.scene.robot = PROCEDURAL_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # IMPORTANT: Must disable replicate_physics for heterogeneous robots
        self.scene.replicate_physics = False
        
        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # No height scan for flat terrain
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        
        # No terrain curriculum
        self.curriculum.terrain_levels = None
        
        # ==================== Actions Configuration ====================
        self.actions.joint_pos.scale = 0.25
        
        # ==================== Events Configuration ====================
        # Disable push for initial training stability
        self.events.randomize_push_robot = None
        
        # Mass randomization
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.5, 1.5)
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "base"
        
        # External force on base
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "base"
        
        # Reset configuration
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
        # General rewards
        self.rewards.is_alive = None
        self.rewards.is_terminated = None
        
        # Tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        
        # Base penalties
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.35  # Adaptive height
        
        # Joint rewards
        self.rewards.joint_torques_l2.weight = -0.0002
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -1.0
        
        # Contact rewards
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(calf|thigh)"
        
        # Feet rewards - use generic foot pattern
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        
        # Standing still
        self.rewards.stand_till.weight = -2.0
        self.rewards.stand_till.params["command_name"] = "base_velocity"
        
        # Body orientation
        self.rewards.body_roll_l2.weight = -3.0
        
        # Disable symmetry rewards (morphology varies)
        self.rewards.joint_symmetry_l2 = None
        self.rewards.action_symmetry_l2 = None
        
        # Disable handstand rewards
        self.rewards.handstand_feet_height_exp = None
        self.rewards.handstand_feet_on_air = None
        self.rewards.handstand_feet_air_time = None
        self.rewards.handstand_orientation_l2 = None
        
        # ==================== Terminations Configuration ====================
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"
        
        # ==================== Commands Configuration ====================
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


# @configclass
# class ProceduralQuadrupedRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
#     """Environment configuration for procedural quadruped on rough terrain.
    
#     This configuration is for training morphology-agnostic policies that can
#     handle rough terrain with varying robot designs.
#     """
    
#     def __post_init__(self):
#         # Post init of parent
#         super().__post_init__()
        
#         # ==================== Scene Configuration ====================
#         # Use procedural quadruped
#         self.scene.robot = PROCEDURAL_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
#         # IMPORTANT: Must disable replicate_physics for heterogeneous robots
#         self.scene.replicate_physics = False
        
#         # Height scanner for procedural robot
#         self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
#         # Reduce terrain complexity for initial training
#         self.scene.terrain.terrain_generator.num_rows = 4
#         self.scene.terrain.terrain_generator.num_cols = 4
        
#         # ==================== Actions Configuration ====================
#         self.actions.joint_pos.scale = 0.25
        
#         # ==================== Events Configuration ====================
#         self.events.randomize_push_robot = None
#         self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.5, 1.5)
#         self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "base"
#         self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "base"
#         self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
#         self.events.randomize_reset_base.params = {
#             "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
#             "velocity_range": {
#                 "x": (0.0, 0.0),
#                 "y": (0.0, 0.0),
#                 "z": (0.0, 0.0),
#                 "roll": (0.0, 0.0),
#                 "pitch": (0.0, 0.0),
#                 "yaw": (0.0, 0.0),
#             },
#         }
        
#         # ==================== Rewards Configuration ====================
#         self.rewards.is_alive = None
#         self.rewards.is_terminated = None
        
#         self.rewards.track_lin_vel_xy_exp.weight = 2.5
#         self.rewards.track_ang_vel_z_exp.weight = 1.0
        
#         self.rewards.lin_vel_z_l2 = None
#         self.rewards.ang_vel_xy_l2 = None
#         self.rewards.flat_orientation_l2 = None
#         self.rewards.base_height_l2 = None
#         self.rewards.body_lin_acc_l2 = None
#         self.rewards.base_ang_vel_x_l2 = None
        
#         self.rewards.joint_torques_l2.weight = -0.0002
#         self.rewards.joint_vel_l1 = None
#         self.rewards.joint_vel_l2 = None
#         self.rewards.joint_acc_l2.weight = -2.5e-7
#         self.rewards.joint_deviation_l1 = None
#         self.rewards.joint_pos_limits = None
#         self.rewards.joint_vel_limits = None
#         self.rewards.applied_torque_limits = None
#         self.rewards.joint_power = None
        
#         self.rewards.action_rate_l2 = None
#         self.rewards.action_l2 = None
        
#         self.rewards.undesired_contacts = None
#         self.rewards.desired_contacts = None
#         self.rewards.contact_forces = None
        
#         self.rewards.feet_air_time.weight = 0.5
#         self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
#         self.rewards.feet_height = None
#         self.rewards.feet_slide.weight = -0.1
#         self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_foot"
        
#         self.rewards.stand_till.weight = -2.0
#         self.rewards.stand_till.params["command_name"] = "base_velocity"
        
#         self.rewards.body_roll_l2 = None
#         self.rewards.body_pitch_l2 = None
        
#         self.rewards.joint_symmetry_l2 = None
#         self.rewards.action_symmetry_l2 = None
        
#         self.rewards.handstand_feet_height_exp = None
#         self.rewards.handstand_feet_on_air = None
#         self.rewards.handstand_feet_air_time = None
#         self.rewards.handstand_orientation_l2 = None
        
#         # ==================== Terminations Configuration ====================
#         self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"
        
#         # ==================== Commands Configuration ====================
#         self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
#         self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
#         self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


# @configclass
# class ProceduralQuadrupedFlatEnvCfg_PLAY(ProceduralQuadrupedFlatEnvCfg):
#     """Play configuration for procedural quadruped on flat terrain."""
    
#     def __post_init__(self):
#         super().__post_init__()
        
#         # Smaller scene for play/evaluation
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
        
#         # Disable observation corruption
#         self.observations.policy.enable_corruption = False
        
#         # Disable randomization events
#         self.events.base_external_force_torque = None
#         self.events.push_robot = None


@configclass
class ProceduralQuadrupedRoughEnvCfg_PLAY(ProceduralQuadrupedRoughEnvCfg):
    """Play configuration for procedural quadruped on rough terrain."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Smaller scene for play/evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        
        # Disable observation corruption
        self.observations.policy.enable_corruption = False
        
        # Disable randomization events
        self.events.base_external_force_torque = None
        self.events.push_robot = None
