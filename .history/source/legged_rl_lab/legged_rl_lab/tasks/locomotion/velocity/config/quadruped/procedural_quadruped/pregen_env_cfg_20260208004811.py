# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0
"""
Pre-Generated Quadruped Environment Configuration.

This module provides environment configurations for training RL agents on
pre-generated quadruped robots loaded from USD files. Unlike the online
procedural approach, this uses offline-generated USD files for:
  - Faster scene initialization (no MjSpec→USD conversion at runtime)
  - Deterministic morphology assignment per environment
  - Morphology parameters injected into observations for adaptive policies

Workflow:
  1. Run batch_generate_usd.py to generate N USD files + manifest.json
  2. Use this config to train with heterogeneous morphologies
  3. Policy receives [obs, morphology_params] and learns morphology-adaptive behavior
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg as ObsTerm
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Import pre-generated USD spawner
from metamorphosis.pregenerated_asset_cfg import PreGeneratedQuadrupedCfg

# Import morphology observation function
from legged_rl_lab.tasks.locomotion.velocity.mdp import observations as obs_mdp


##
# Default USD directory (relative to this file)
# Users should generate USDs here or override in the config
##
_DEFAULT_USD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "..", "..", "..",
    "metamorphosis", "generated_quadrupeds"
)


##
# Pre-Generated Quadruped ArticulationCfg
##

PREGENERATED_QUADRUPED_CFG = ArticulationCfg(
    spawn=PreGeneratedQuadrupedCfg(
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        usd_dir=_DEFAULT_USD_DIR,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            ".*_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        pos=(0, 0, 0.8),  # High initial position for varied leg lengths
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
class PreGenQuadrupedFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Environment configuration for pre-generated quadrupeds on flat terrain.
    
    Key differences from online ProceduralQuadrupedFlatEnvCfg:
      - Uses PreGeneratedQuadrupedCfg spawner (loads USD files, no runtime generation)
      - Adds morphology_params observation term (7-dim per env)
      - Observation dim = base(60) + morphology(7) = 67
      - Better training stability due to cached morphology info
    """
    
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # ==================== Scene Configuration ====================
        self.scene.robot = PREGENERATED_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
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
        
        # ==================== Observations Configuration ====================
        # 🔑 Add morphology parameters to observation space
        # This gives the policy information about the robot's body shape,
        # enabling morphology-adaptive behavior.
        self.observations.policy.morphology_params = ObsTerm(
            func=obs_mdp.morphology_params,
            scale=1.0,
            clip=(-3.0, 3.0),
        )
        
        # ==================== Actions Configuration ====================
        self.actions.joint_pos.scale = 0.25
        
        # ==================== Events Configuration ====================
        # Disable push for initial training stability
        self.events.randomize_push_robot = None
        
        # Mass randomization - smaller range since morphology already varies
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.3, 0.8)
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "base"
        
        # External force on base
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "base"
        
        # Reset configuration
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        
        # ==================== Rewards Configuration ====================
        # General rewards
        self.rewards.is_alive = None
        self.rewards.is_terminated = None
        
        # ===== Tracking Rewards (main learning signal) =====
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        
        # ===== Base Penalties =====
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.body_lin_acc_l2 = None
        self.rewards.base_ang_vel_x_l2 = None
        
        # ===== Joint Penalties =====
        self.rewards.joint_torques_l2.weight = -1.0e-5
        self.rewards.joint_vel_l1 = None
        self.rewards.joint_vel_l2.weight = -0.005
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_deviation_l1 = None
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.joint_vel_limits = None
        self.rewards.applied_torque_limits = None
        self.rewards.joint_power = None
        
        # ===== Action Penalties =====
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.action_l2 = None
        
        # ===== Contact Rewards =====
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(calf|thigh)"
        self.rewards.desired_contacts = None
        self.rewards.contact_forces = None
        
        # ===== Feet Rewards =====
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_height = None
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_slide.params["asset_cfg"] = SceneEntityCfg("robot", body_names=".*_foot")
        
        # ===== Other Rewards =====
        self.rewards.stand_till.weight = -0.2
        self.rewards.stand_till.params["command_name"] = "base_velocity"
        self.rewards.body_roll_l2.weight = -2.0
        self.rewards.body_pitch_l2 = None
        
        # Symmetry - disabled for heterogeneous training
        self.rewards.joint_symmetry_l2 = None
        self.rewards.action_symmetry_l2 = None
        
        # Handstand - not applicable
        self.rewards.handstand_feet_height_exp = None
        self.rewards.handstand_feet_on_air = None
        self.rewards.handstand_feet_air_time = None
        self.rewards.handstand_orientation_l2 = None
        
        # ==================== Terminations ====================
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"
        
        # ==================== Commands ====================
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class PreGenQuadrupedFlatEnvCfg_PLAY(PreGenQuadrupedFlatEnvCfg):
    """Play configuration for pre-generated quadruped on flat terrain."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Smaller scene for play/evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable observation corruption
        self.observations.policy.enable_corruption = False
        
        # Disable randomization events
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
