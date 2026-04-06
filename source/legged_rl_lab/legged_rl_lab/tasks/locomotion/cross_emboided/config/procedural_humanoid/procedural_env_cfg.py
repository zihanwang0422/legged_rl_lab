# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Procedural Humanoid Environment Configuration.

Flat and rough terrain environment configurations for procedurally generated
humanoid (biped) robots with varying torso/leg dimensions.
All classes inherit from ``CrossEmbodiedLocomotionEnvCfg`` which provides the
cross-embodied specific defaults (events, rewards, observations, actions).
"""

import numpy as np

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm

import isaaclab.sim as sim_utils

from legged_rl_lab.tasks.locomotion.cross_emboided.cross_emboided_env_cfg import (
    CrossEmbodiedLocomotionEnvCfg,
)
from legged_rl_lab.tasks.locomotion.cross_emboided.mdp import morphology_params

from metamorphosis.asset_cfg import ProceduralBipedCfg  # isort: skip


PROCEDURAL_HUMANOID_CFG = ArticulationCfg(
    spawn=ProceduralBipedCfg(
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        torso_link_length_range=(0.10, 0.16),
        torso_link_width_range=(0.18, 0.26),
        torso_link_height_range=(0.08, 0.14),
        pelvis_height_range=(0.05, 0.08),
        hip_spacing_range=(0.16, 0.24),
        hip_pitch_link_length_range=(0.03, 0.06),
        hip_pitch_link_radius_range=(0.02, 0.04),
        hip_roll_link_length_range=(0.03, 0.06),
        hip_roll_link_radius_range=(0.02, 0.04),
        hip_pitch_link_initroll_range=(0.0, np.pi / 2),
        leg_length_range=(0.5, 0.7),
        shin_ratio_range=(0.85, 1.15),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.2),
        joint_pos={
            ".*_hip_pitch_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=300.0,
            velocity_limit=30.0,
            stiffness=80.0,
            damping=2.0,
            armature=0.01,
            friction=0.05,
        ),
    },
)


@configclass
class ProceduralHumanoidFlatEnvCfg(CrossEmbodiedLocomotionEnvCfg):
    """Flat terrain environment for procedurally generated humanoids."""

    def __post_init__(self):
        super().__post_init__()

        # ====Scene Cfg====
        self.scene.robot = PROCEDURAL_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ====Terrain Cfg====
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        # Morphology parameters observation (procedural-specific)
        self.observations.policy.morphology_params = ObsTerm(func=morphology_params)

        # ====Event Cfg====
        # Flat terrain: disable push and COM randomisation
        self.events.randomize_push_robot = None
        self.events.randomize_com_positions = None
        # Humanoid body name for mass/force randomisation
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        # ====Rewards: humanoid-specific body names====
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(hip|knee|pelvis).*"
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*_ankle_roll_link"

        # Left-right gait symmetry
        self.rewards.joint_symmetry_l2.weight = -0.1
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
            ["left_.*_joint", "right_.*_joint"],
        ]
        self.rewards.action_symmetry_l2.weight = -0.05
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [
            ["left_.*_joint", "right_.*_joint"],
        ]

        # ====Termination Cfg====
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "(torso_link|pelvis)"


@configclass
class ProceduralHumanoidFlatEnvCfg_PLAY(ProceduralHumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None


@configclass
class ProceduralHumanoidRoughEnvCfg(CrossEmbodiedLocomotionEnvCfg):
    """Rough terrain environment for procedurally generated humanoids."""

    def __post_init__(self):
        super().__post_init__()

        # ====Scene Cfg====
        self.scene.robot = PROCEDURAL_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Morphology parameters observation (procedural-specific)
        self.observations.policy.morphology_params = ObsTerm(func=morphology_params)
        self.observations.policy.height_scan.scale = 1.0

        # ====Event Cfg====
        # Humanoid body name for mass/force randomisation
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        # ====Rewards: humanoid-specific body names====
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(hip|knee|pelvis).*"
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*_ankle_roll_link"

        # Left-right gait symmetry
        self.rewards.joint_symmetry_l2.weight = -0.1
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
            ["left_.*_joint", "right_.*_joint"],
        ]
        self.rewards.action_symmetry_l2.weight = -0.05
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [
            ["left_.*_joint", "right_.*_joint"],
        ]

        # ====Termination Cfg====
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "(torso_link|pelvis)"


@configclass
class ProceduralHumanoidRoughEnvCfg_PLAY(ProceduralHumanoidRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None
