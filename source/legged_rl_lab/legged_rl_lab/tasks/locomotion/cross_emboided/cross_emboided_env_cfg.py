# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Standalone base environment configuration for cross-embodied locomotion tasks.

This module is completely independent of any velocity-task module.
It defines all scene / MDP config dataclasses from scratch and inherits
directly from ``ManagerBasedRLEnvCfg``.

All cross-embodied task configs should inherit from
``CrossEmbodiedLocomotionEnvCfg`` defined here.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_rl_lab.tasks.locomotion.cross_emboided.mdp as mdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##############################################################################
# Scene
##############################################################################


@configclass
class CrossEmbodiedSceneCfg(InteractiveSceneCfg):
    """Terrain scene for cross-embodied locomotion tasks."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Robot is set by child configs
    robot: ArticulationCfg = MISSING

    # Height scanner — prim_path overridden per robot in child configs
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##############################################################################
# Commands
##############################################################################


@configclass
class CrossEmbodiedCommandsCfg:
    """Velocity command specification for cross-embodied tasks."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


##############################################################################
# Actions
##############################################################################


@configclass
class CrossEmbodiedActionsCfg:
    """Action specification for cross-embodied tasks (joint-position only)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )
    # joint_vel intentionally absent — cross-embodied uses position control only


##############################################################################
# Observations
##############################################################################


@configclass
class CrossEmbodiedObservationsCfg:
    """Observation specifications for cross-embodied locomotion."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy (actor) observations."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=1.0,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100.0, 100.0),
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            scale=1.0,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(
            func=mdp.last_action,
            scale=1.0,
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic (value function) observations — privileged information."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##############################################################################
# Events (Domain Randomisation)
##############################################################################


@configclass
class CrossEmbodiedEventCfg:
    """Domain-randomisation events for cross-embodied locomotion tasks.

    All event term names use the ``randomize_*`` prefix so child configs can
    zero out individual terms by assigning ``None``.
    """

    # ── startup ─────────────────────────────────────────────────────────────

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # ── reset ────────────────────────────────────────────────────────────────

    randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ── interval ─────────────────────────────────────────────────────────────

    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


##############################################################################
# Rewards
##############################################################################


@configclass
class CrossEmbodiedRewardsCfg:
    """Reward terms for cross-embodied locomotion tasks.

    Weights are set to cross-embodied defaults.  Child configs override
    body-name-specific terms (undesired_contacts, feet_air_time, etc.) and
    morphology-specific symmetry terms.
    """

    # ── velocity tracking ────────────────────────────────────────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ── base penalties ───────────────────────────────────────────────────────
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # ── joint penalties ──────────────────────────────────────────────────────
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    # ── action penalties ─────────────────────────────────────────────────────
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # ── contact / feet ───────────────────────────────────────────────────────
    # body_names must be overridden by child configs
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"),
            "threshold": 1.0,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    # ── standing still ───────────────────────────────────────────────────────
    stand_till = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )

    # ── body orientation ─────────────────────────────────────────────────────
    body_roll_l2 = RewTerm(
        func=mdp.body_roll_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── symmetry (body-names set by child configs) ───────────────────────────
    joint_symmetry_l2 = RewTerm(
        func=mdp.joint_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [[".*L_hip_joint"], [".*R_hip_joint"]],
        },
    )
    action_symmetry_l2 = RewTerm(
        func=mdp.action_mirror,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [[".*L_hip_joint"], [".*R_hip_joint"]],
        },
    )


##############################################################################
# Terminations
##############################################################################


@configclass
class CrossEmbodiedTerminationsCfg:
    """Termination conditions for cross-embodied locomotion."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


##############################################################################
# Curriculum
##############################################################################


@configclass
class CrossEmbodiedCurriculumCfg:
    """Curriculum terms for cross-embodied locomotion."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##############################################################################
# Base Environment Configuration
##############################################################################


@configclass
class CrossEmbodiedLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Standalone base configuration for all cross-embodied locomotion tasks.

    Inherits directly from ``ManagerBasedRLEnvCfg`` — no dependency on any
    velocity-task module.

    Child configs must override at minimum:
    - ``scene.robot`` — the specific robot articulation.
    - ``scene.height_scanner.prim_path`` — scanner body attachment point.
    - Event body names (e.g. ``events.randomize_rigid_body_mass_base.params``).
    - Reward body-name params (``undesired_contacts``, ``feet_air_time``, etc.).
    - Symmetry reward ``mirror_joints`` and weights.
    - ``terminations.illegal_contact`` body name.
    - ``observations.policy.morphology_params`` (for procedural robots).
    """

    scene: CrossEmbodiedSceneCfg = CrossEmbodiedSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: CrossEmbodiedObservationsCfg = CrossEmbodiedObservationsCfg()
    actions: CrossEmbodiedActionsCfg = CrossEmbodiedActionsCfg()
    commands: CrossEmbodiedCommandsCfg = CrossEmbodiedCommandsCfg()
    rewards: CrossEmbodiedRewardsCfg = CrossEmbodiedRewardsCfg()
    terminations: CrossEmbodiedTerminationsCfg = CrossEmbodiedTerminationsCfg()
    events: CrossEmbodiedEventCfg = CrossEmbodiedEventCfg()
    curriculum: CrossEmbodiedCurriculumCfg = CrossEmbodiedCurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialisation."""
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Heterogeneous morphologies require independent physics sim
        self.scene.replicate_physics = False

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Terrain curriculum sync
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

