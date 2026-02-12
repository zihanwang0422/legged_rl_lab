# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Procedural Quadruped Environment Configuration.

Flat terrain environment configuration for procedurally generated quadruped
robots with varying body/leg dimensions but consistent joint topology.
"""

import numpy as np

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg

import isaaclab.sim as sim_utils

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from metamorphosis.asset_cfg import ProceduralQuadrupedCfg  # isort: skip


# ============================================================
# Procedural Quadruped ArticulationCfg
# ============================================================

PROCEDURAL_QUADRUPED_CFG = ArticulationCfg(
    spawn=ProceduralQuadrupedCfg(
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        base_length_range=(0.5, 1.0),
        base_width_range=(0.3, 0.4),
        base_height_range=(0.15, 0.25),
        leg_length_range=(0.4, 0.8),
        calf_length_ratio=(0.9, 1.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # Initial value, overridden per-robot during spawn and modify_articulation
        joint_pos={
            ".*": 0.0,  # Placeholder, overridden per-robot by QuadrupedBuilder.modify_articulation
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1000.0,
            velocity_limit=30.0,
            stiffness=200.0,
            damping=2.0,
            armature=0.01,
            friction=0.01,
        ),
    },
)


# ============================================================
# Flat Environment Configuration
# ============================================================

@configclass
class ProceduralQuadrupedFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Flat terrain environment for procedurally generated quadrupeds."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ====Scene Cfg====
        self.scene.robot = PROCEDURAL_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # procedural robots are non-homogeneous, must disable replicate_physics
        self.scene.replicate_physics = False

        # ====Terrain Cfg====
        # flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        
        # # Add morphology parameters as observation for heterogeneous training
        # from isaaclab.managers import ObservationTermCfg as ObsTerm
        # from legged_rl_lab.tasks.locomotion.velocity.mdp import observations
        # self.observations.policy.morphology_params = ObsTerm(func=observations.morphology_params)

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # ====Action Cfg====
        self.actions.joint_pos.scale = 0.25
        # procedural quadrupeds have wider joint ranges, remove Go1-specific clips
        self.actions.joint_pos.clip = None
        # Only use position control for procedural robots (disable velocity control)
        self.actions.joint_vel = None

        # ====Event Cfg====
        self.events.randomize_push_robot = None
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "base"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "base"
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
        self.events.randomize_com_positions = None

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

        # Disable base_height_l2: each robot has different standing height
        self.rewards.base_height_l2 = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.base_ang_vel_x_l2 = None

        # ===== Joint Rewards =====
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l1 = None
        self.rewards.joint_vel_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.0e-7
        self.rewards.joint_deviation_l1 = None
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_vel_limits = None
        self.rewards.applied_torque_limits = None
        self.rewards.joint_power = None

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_l2 = None

        # ===== Contact Rewards =====
        # Undesired contacts
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(calf|thigh)"
        self.rewards.desired_contacts = None
        self.rewards.contact_forces = None

        # Feet rewards
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_height = None
        self.rewards.feet_slide.weight = -0.1

        # ===== Other/Custom Rewards =====
        # Standing still
        self.rewards.stand_till.weight = -0.5
        self.rewards.stand_till.params["command_name"] = "base_velocity"

        # Body orientation
        self.rewards.body_roll_l2.weight = -5.0
        self.rewards.body_pitch_l2 = None

        # Diagonal gait symmetry (Trot Gait: FL+RR, FR+RL)
        self.rewards.joint_symmetry_l2.weight = -0.1
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
            ["FL_.*_joint", "RR_.*_joint"],  # 对角线1
            ["FR_.*_joint", "RL_.*_joint"],  # 对角线2
        ]
        self.rewards.action_symmetry_l2.weight = -0.05
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [
            ["FL_.*_joint", "RR_.*_joint"],  # 对角线1
            ["FR_.*_joint", "RL_.*_joint"],  # 对角线2
        ]

        # ===== Handstand Rewards (disabled) =====
        self.rewards.handstand_feet_height_exp = None
        self.rewards.handstand_feet_on_air = None
        self.rewards.handstand_feet_air_time = None
        self.rewards.handstand_orientation_l2 = None

        # ====Termination Cfg====
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"

        # ====Commands Cfg====
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class ProceduralQuadrupedFlatEnvCfg_PLAY(ProceduralQuadrupedFlatEnvCfg):
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
        self.events.randomize_push_robot = None


@configclass
class ProceduralQuadrupedRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough terrain environment for procedurally generated quadrupeds."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ====Scene Cfg====
        self.scene.robot = PROCEDURAL_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # procedural robots are non-homogeneous
        self.scene.replicate_physics = False
        
        # Add morphology parameters as observation
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from legged_rl_lab.tasks.locomotion.velocity.mdp import observations
        self.observations.policy.morphology_params = ObsTerm(func=observations.morphology_params)

        # ====Action Cfg====
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = None

        # ====Event Cfg====
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "base"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        # ==================== Rewards Configuration ====================

        # ===== General Rewards =====
        self.rewards.is_alive = None
        self.rewards.is_terminated = None

        # ===== Base Rewards =====
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.base_height_l2 = None  # Each robot has different standing height
        self.rewards.body_lin_acc_l2 = None
        self.rewards.base_ang_vel_x_l2 = None

        # ===== Joint Rewards =====
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l1 = None
        self.rewards.joint_vel_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.0e-7
        self.rewards.joint_deviation_l1 = None
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_vel_limits = None
        self.rewards.applied_torque_limits = None
        self.rewards.joint_power = None

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_l2 = None

        # ===== Contact Rewards =====
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(calf|thigh)"
        self.rewards.desired_contacts = None
        self.rewards.contact_forces = None

        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_height = None
        self.rewards.feet_slide.weight = -0.1

        # ===== Other/Custom Rewards =====
        self.rewards.stand_till.weight = -0.5
        self.rewards.stand_till.params["command_name"] = "base_velocity"
        self.rewards.body_roll_l2.weight = -5.0
        self.rewards.body_pitch_l2 = None

        self.rewards.joint_symmetry_l2.weight = -0.1
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
            ["FL_.*_joint", "RR_.*_joint"],
            ["FR_.*_joint", "RL_.*_joint"],
        ]
        self.rewards.action_symmetry_l2.weight = -0.05
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [
            ["FL_.*_joint", "RR_.*_joint"],
            ["FR_.*_joint", "RL_.*_joint"],
        ]

        self.rewards.handstand_feet_height_exp = None
        self.rewards.handstand_feet_on_air = None
        self.rewards.handstand_feet_air_time = None
        self.rewards.handstand_orientation_l2 = None

        # ====Termination Cfg====
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"

        # ====Commands Cfg====
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class ProceduralQuadrupedRoughEnvCfg_PLAY(ProceduralQuadrupedRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None
