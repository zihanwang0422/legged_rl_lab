# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Cross-embodied Mixed Procedural (Biped + Quadruped) Environment Configuration.

Scene layout
------------
Both a procedural biped (26 DOF) and a procedural quadruped (12 DOF) are
spawned in **every** environment instance.  The first ``floor(N * humanoid_ratio)``
envs make the biped active (quad is parked 100 m above ground) and the rest
do the opposite.

Observations (actor, flat: 98 dims)
------------------------------------
ang_vel(3) + proj_grav(3) + cmd(3) + joint_pos(26) + joint_vel(26) + last_action(26)
 + morphology(11) = 98.
For quad envs joint dims 12-25 are zero-padded.

Critic observations (101 dims)
-------------------------------
Adds base_lin_vel(3) = 101.

Actions (26 dims)
-----------------
Unified across both morphologies.  Quad envs use only the first 12 dims.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_rl_lab.tasks.locomotion.velocity.mdp as mdp
from legged_rl_lab.tasks.locomotion.cross_emboided import mdp as cross_mdp
from legged_rl_lab.tasks.locomotion.cross_emboided.mdp.cross_procedural_mdp import (
    ProceduralMixedJointPosActionCfg,
)
from legged_rl_lab.tasks.locomotion.cross_emboided.mdp.procedural_obs import morphology_params

# Pre-defined procedural articulation configs (imported from single-embodiment cfgs)
from legged_rl_lab.tasks.locomotion.cross_emboided.config.procedural_humanoid.humanoid_env_cfg import (
    PROCEDURAL_HUMANOID_CFG,
)
from legged_rl_lab.tasks.locomotion.cross_emboided.config.procedural_quadruped.quadruped_env_cfg import (
    PROCEDURAL_QUADRUPED_CFG,
)


##############################################################################
# Scene
##############################################################################


@configclass
class ProceduralMixedSceneCfg(InteractiveSceneCfg):
    """Scene with both procedural biped and quadruped in every env instance."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Biped (26 DOF) – present in every env; parked in quad envs.
    robot: ArticulationCfg = PROCEDURAL_HUMANOID_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Quadruped (12 DOF) – present in every env; parked in biped envs.
    quad_robot: ArticulationCfg = PROCEDURAL_QUADRUPED_CFG.replace(
        prim_path="{ENV_REGEX_NS}/QuadRobot",
    )

    # Contact sensors (all bodies, for reward/termination dispatch)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    contact_forces_quad = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/QuadRobot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##############################################################################
# Observations
##############################################################################


@configclass
class ProceduralMixedObsCfg:
    """Observation specifications for the mixed biped+quad environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """98-dim actor observations (flat terrain, no height scan)."""

        base_ang_vel = ObsTerm(
            func=cross_mdp.base_ang_vel_mixed,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=cross_mdp.projected_gravity_mixed,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=cross_mdp.joint_pos_mixed,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=cross_mdp.joint_vel_mixed,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=cross_mdp.last_action_mixed)
        morphology_params = ObsTerm(func=morphology_params)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """101-dim critic observations (adds base_lin_vel)."""

        base_lin_vel = ObsTerm(func=cross_mdp.base_lin_vel_mixed)
        base_ang_vel = ObsTerm(func=cross_mdp.base_ang_vel_mixed, scale=0.2)
        projected_gravity = ObsTerm(func=cross_mdp.projected_gravity_mixed)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=cross_mdp.joint_pos_mixed)
        joint_vel = ObsTerm(func=cross_mdp.joint_vel_mixed, scale=0.05)
        last_action = ObsTerm(func=cross_mdp.last_action_mixed)
        morphology_params = ObsTerm(func=morphology_params)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


##############################################################################
# Actions
##############################################################################


@configclass
class ProceduralMixedActionsCfg:
    """Unified 26-dim joint-position actions."""

    joint_pos: ProceduralMixedJointPosActionCfg = ProceduralMixedJointPosActionCfg(
        scale_biped=0.5,
        scale_quad=0.25,
    )


##############################################################################
# Commands
##############################################################################


@configclass
class ProceduralMixedCommandsCfg:
    """Velocity commands.

    ``asset_name`` points to ``robot`` (biped).  For quad envs the heading
    metric uses the parked biped's pose (harmless — only affects logging).
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


##############################################################################
# Events
##############################################################################


@configclass
class ProceduralMixedEventsCfg:
    """Domain randomisation for the mixed env.

    Reset events apply to BOTH robots; the active robot uses the randomised
    pose while the inactive robot is immediately re-parked.
    """

    # ── startup ────────────────────────────────────────────────────────────

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
    physics_material_quad = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("quad_robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_biped_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    add_quad_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "asset_cfg": SceneEntityCfg("quad_robot", body_names="base"),
        },
    )

    # ── reset ────────────────────────────────────────────────────────────

    reset_biped_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_biped_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_quad_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("quad_robot"),
        },
    )
    reset_quad_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("quad_robot"),
        },
    )


##############################################################################
# Rewards
##############################################################################


@configclass
class ProceduralMixedRewardsCfg:
    """Reward terms for the mixed biped + quadruped policy.

    All rewards use ``*_mixed`` dispatch functions that automatically select
    the active robot per environment via ``env.is_humanoid_env``.
    """

    # ── velocity tracking ────────────────────────────────────────────────
    track_lin_vel_xy = RewTerm(
        func=cross_mdp.track_lin_vel_xy_exp_mixed,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=cross_mdp.track_ang_vel_z_exp_mixed,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ── base penalties ───────────────────────────────────────────────────
    ang_vel_xy_l2 = RewTerm(func=cross_mdp.ang_vel_xy_l2_mixed, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=cross_mdp.flat_orientation_l2_mixed, weight=-2.0)
    body_roll_l2 = RewTerm(func=cross_mdp.body_roll_l2_mixed, weight=-5.0)

    # ── joint / action penalties ─────────────────────────────────────────
    joint_torques = RewTerm(func=cross_mdp.joint_torques_l2_mixed, weight=-2.0e-6)
    joint_vel = RewTerm(func=cross_mdp.joint_vel_l2_mixed, weight=-0.001)
    joint_acc = RewTerm(func=cross_mdp.joint_acc_l2_mixed, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=cross_mdp.joint_pos_limits_mixed, weight=-2.0)
    action_rate = RewTerm(func=cross_mdp.action_rate_l2_mixed, weight=-0.005)

    # ── standing still ───────────────────────────────────────────────────
    stand_still = RewTerm(
        func=cross_mdp.stand_still_mixed,
        weight=-0.5,
        params={"command_name": "base_velocity", "command_threshold": 0.06},
    )

    # ── contact / feet ───────────────────────────────────────────────────
    undesired_contacts = RewTerm(
        func=cross_mdp.undesired_contacts_mixed,
        weight=-1.0,
    )
    feet_air_time = RewTerm(
        func=cross_mdp.feet_air_time_mixed,
        weight=0.5,
        params={"threshold": 0.4, "command_name": "base_velocity"},
    )
    feet_slide = RewTerm(
        func=cross_mdp.feet_slide_mixed,
        weight=-0.1,
    )


##############################################################################
# Terminations
##############################################################################


@configclass
class ProceduralMixedTerminationsCfg:
    """Termination terms for the mixed env."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact_base = DoneTerm(
        func=cross_mdp.illegal_contact_base_mixed,
        params={"threshold": 1.0},
    )


##############################################################################
# Curriculum
##############################################################################


@configclass
class ProceduralMixedCurriculumCfg:
    pass


##############################################################################
# Environment Configurations
##############################################################################


@configclass
class ProceduralMixedFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Mixed biped+quad on flat terrain."""

    scene: ProceduralMixedSceneCfg = ProceduralMixedSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    observations: ProceduralMixedObsCfg = ProceduralMixedObsCfg()
    actions: ProceduralMixedActionsCfg = ProceduralMixedActionsCfg()
    commands: ProceduralMixedCommandsCfg = ProceduralMixedCommandsCfg()
    rewards: ProceduralMixedRewardsCfg = ProceduralMixedRewardsCfg()
    terminations: ProceduralMixedTerminationsCfg = ProceduralMixedTerminationsCfg()
    events: ProceduralMixedEventsCfg = ProceduralMixedEventsCfg()
    curriculum: ProceduralMixedCurriculumCfg = ProceduralMixedCurriculumCfg()

    # Fraction of envs running biped (rest run quadruped)
    humanoid_ratio: float = 0.5

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Heterogeneous morphologies require independent physics sim
        self.scene.replicate_physics = False


@configclass
class ProceduralMixedFlatEnvCfg_PLAY(ProceduralMixedFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class ProceduralMixedRoughEnvCfg(ProceduralMixedFlatEnvCfg):
    """Mixed biped+quad on rough terrain (extends flat cfg)."""

    def __post_init__(self):
        super().__post_init__()
        # Switch to generated rough terrain
        from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 1


@configclass
class ProceduralMixedRoughEnvCfg_PLAY(ProceduralMixedRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.observations.policy.enable_corruption = False
