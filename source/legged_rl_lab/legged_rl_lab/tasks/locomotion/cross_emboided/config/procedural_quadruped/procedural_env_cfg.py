# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Procedural Quadruped Environment Configuration.

Flat and rough terrain environment configurations for procedurally generated
quadruped robots with varying body/leg dimensions but consistent joint topology.
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
class ProceduralQuadrupedFlatEnvCfg(CrossEmbodiedLocomotionEnvCfg):
    """Flat terrain environment for procedurally generated quadrupeds."""

    def __post_init__(self):
        super().__post_init__()

        # ====Scene Cfg====
        self.scene.robot = PROCEDURAL_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        # Quadruped body name for mass/force randomisation
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "base"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        # ====Rewards: quadruped-specific body names====
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(calf|thigh)"
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"

        # Diagonal gait symmetry (Trot: FL+RR, FR+RL)
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

        # ====Termination Cfg====
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"


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
class ProceduralQuadrupedRoughEnvCfg(CrossEmbodiedLocomotionEnvCfg):
    """Rough terrain environment for procedurally generated quadrupeds."""

    def __post_init__(self):
        super().__post_init__()

        # ====Scene Cfg====
        self.scene.robot = PROCEDURAL_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Morphology parameters observation (procedural-specific)
        self.observations.policy.morphology_params = ObsTerm(func=morphology_params)
        self.observations.policy.height_scan.scale = 1.0

        # ====Event Cfg====
        # Quadruped body name for mass/force randomisation
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "base"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        # ====Rewards: quadruped-specific body names====
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(calf|thigh)"
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"

        # Diagonal gait symmetry (Trot: FL+RR, FR+RL)
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

        # ====Termination Cfg====
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"


@configclass
class ProceduralQuadrupedRoughEnvCfg_PLAY(ProceduralQuadrupedRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None
