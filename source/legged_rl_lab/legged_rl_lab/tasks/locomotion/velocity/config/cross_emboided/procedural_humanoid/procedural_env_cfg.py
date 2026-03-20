# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Procedural Humanoid Environment Configuration.

Flat and rough terrain environment configurations for procedurally generated
humanoid (biped) robots with varying torso/leg dimensions.
"""

import numpy as np

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab.sim as sim_utils

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

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
class ProceduralHumanoidFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Flat terrain environment for procedurally generated humanoids."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PROCEDURAL_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.replicate_physics = False

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from legged_rl_lab.tasks.locomotion.velocity.mdp import observations

        self.observations.policy.morphology_params = ObsTerm(func=observations.morphology_params)

        self.observations.policy.base_lin_vel.scale = 0.2
        self.observations.policy.base_lin_vel.clip = (-100.0, 100.0)
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.base_ang_vel.clip = (-100.0, 100.0)
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.projected_gravity.clip = (-100.0, 100.0)
        self.observations.policy.velocity_commands.clip = (-100.0, 100.0)
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_pos.clip = (-100.0, 100.0)
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_vel.clip = (-100.0, 100.0)
        self.observations.policy.actions.scale = 1.0
        self.observations.policy.actions.clip = (-100.0, 100.0)

        self.curriculum.terrain_levels = None

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = None
        self.actions.joint_vel = None

        self.events.randomize_push_robot = None
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "torso_link"
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

        self.rewards.is_alive = None
        self.rewards.is_terminated = None

        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.base_height_l2 = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.base_ang_vel_x_l2 = None

        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l1 = None
        self.rewards.joint_vel_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.0e-7
        self.rewards.joint_deviation_l1 = None
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_vel_limits = None
        self.rewards.applied_torque_limits = None
        self.rewards.joint_power = None

        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_l2 = None

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(hip|knee|pelvis).*"
        self.rewards.desired_contacts = None
        self.rewards.contact_forces = None

        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_height = None
        self.rewards.feet_slide.weight = -0.1

        self.rewards.stand_till.weight = -0.5
        self.rewards.stand_till.params["command_name"] = "base_velocity"

        self.rewards.body_roll_l2.weight = -5.0
        self.rewards.body_pitch_l2 = None

        self.rewards.joint_symmetry_l2.weight = -0.1
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
            ["left_.*_joint", "right_.*_joint"],
        ]
        self.rewards.action_symmetry_l2.weight = -0.05
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [
            ["left_.*_joint", "right_.*_joint"],
        ]

        self.rewards.handstand_feet_height_exp = None
        self.rewards.handstand_feet_on_air = None
        self.rewards.handstand_feet_air_time = None
        self.rewards.handstand_orientation_l2 = None

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "(torso_link|pelvis)"

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


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
class ProceduralHumanoidRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough terrain environment for procedurally generated humanoids."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PROCEDURAL_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.replicate_physics = False

        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from legged_rl_lab.tasks.locomotion.velocity.mdp import observations

        self.observations.policy.morphology_params = ObsTerm(func=observations.morphology_params)

        self.observations.policy.base_lin_vel.scale = 0.2
        self.observations.policy.base_lin_vel.clip = (-100.0, 100.0)
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.base_ang_vel.clip = (-100.0, 100.0)
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.projected_gravity.clip = (-100.0, 100.0)
        self.observations.policy.velocity_commands.clip = (-100.0, 100.0)
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_pos.clip = (-100.0, 100.0)
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_vel.clip = (-100.0, 100.0)
        self.observations.policy.actions.scale = 1.0
        self.observations.policy.actions.clip = (-100.0, 100.0)
        self.observations.policy.height_scan.scale = 1.0

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = None

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        self.rewards.is_alive = None
        self.rewards.is_terminated = None

        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.base_height_l2 = None
        self.rewards.body_lin_acc_l2 = None
        self.rewards.base_ang_vel_x_l2 = None

        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l1 = None
        self.rewards.joint_vel_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.0e-7
        self.rewards.joint_deviation_l1 = None
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_vel_limits = None
        self.rewards.applied_torque_limits = None
        self.rewards.joint_power = None

        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_l2 = None

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(hip|knee|pelvis).*"
        self.rewards.desired_contacts = None
        self.rewards.contact_forces = None

        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_ankle_roll_link"
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_height = None
        self.rewards.feet_slide.weight = -0.1

        self.rewards.stand_till.weight = -0.5
        self.rewards.stand_till.params["command_name"] = "base_velocity"
        self.rewards.body_roll_l2.weight = -5.0
        self.rewards.body_pitch_l2 = None

        self.rewards.joint_symmetry_l2.weight = -0.1
        self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
            ["left_.*_joint", "right_.*_joint"],
        ]
        self.rewards.action_symmetry_l2.weight = -0.05
        self.rewards.action_symmetry_l2.params["mirror_joints"] = [
            ["left_.*_joint", "right_.*_joint"],
        ]

        self.rewards.handstand_feet_height_exp = None
        self.rewards.handstand_feet_on_air = None
        self.rewards.handstand_feet_air_time = None
        self.rewards.handstand_orientation_l2 = None

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "(torso_link|pelvis)"

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class ProceduralHumanoidRoughEnvCfg_PLAY(ProceduralHumanoidRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.events.randomize_push_robot = None
