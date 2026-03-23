# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Cross-embodied G1 + Go2 flat locomotion environment.

Scene layout
------------
Both Unitree G1 (29 DOF) and Unitree Go2 (12 DOF) are spawned in **every**
environment instance.  The first ``n // 2`` envs make G1 the active robot
(Go2 is parked 100 m above ground) and the remaining envs do the opposite.

At each reset, the active robot is reset to a randomised starting pose while
the inactive robot is re-parked at altitude.

Observations (actor, 98 dims total, history=1)
----------------------------------------------
2 + 3 + 3 + 3 + 29 + 29 + 29 = 98

    [robot_id(2) | ang_vel(3) | proj_grav(3) | cmd(3)
     | joint_pos(29) | joint_vel(29) | last_action(29)]

For Go2 envs the joint dims 12-28 are zero-padded.

Critic observations (101 dims)
-------------------------------
2 + 3 + 3 + 3 + 3 + 29 + 29 + 29 = 101  (adds base_lin_vel)

Actions (29 dims)
-----------------
Unified across both robots.  Go2 envs use only the first 12 dims.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import legged_rl_lab.tasks.locomotion.velocity.mdp as mdp
from legged_rl_lab.assets.unitree import UNITREE_G1_29DOF_CFG, UNITREE_GO2_CFG
from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import CommandsCfg

from . import mdp as cross_mdp
from .mdp.cross_embodied_mdp import CrossEmbodiedJointPosActionCfg


##############################################################################
# Scene
##############################################################################


@configclass
class CrossEmbodiedG1Go2SceneCfg(InteractiveSceneCfg):
    """Scene with both G1 and Go2 in every environment instance."""

    # Flat ground plane
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

    # G1 (29 DOF) – present in every env; parked in Go2 envs.
    robot_g1: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/G1",
    )

    # Go2 (12 DOF) – present in every env; parked in G1 envs.
    robot_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Go2",
    )

    # Contact sensors (all bodies, used for termination / reward)
    contact_forces_g1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/G1/.*",
        history_length=3,
        track_air_time=True,
    )
    contact_forces_go2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/.*",
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
# Observations  (history=1, 98 policy dims / 101 critic dims)
##############################################################################


@configclass
class CrossEmbodiedObsCfg:
    """Observation specifications for the cross-embodied env."""

    @configclass
    class PolicyCfg(ObsGroup):
        """98-dim actor observations."""

        robot_id = ObsTerm(func=cross_mdp.robot_type_id)
        base_ang_vel = ObsTerm(
            func=cross_mdp.base_ang_vel_cross,
            scale=0.2,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=cross_mdp.projected_gravity_cross,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=cross_mdp.joint_pos_cross,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=cross_mdp.joint_vel_cross,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=cross_mdp.last_action_cross)

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """101-dim critic observations (adds base_lin_vel)."""

        robot_id = ObsTerm(func=cross_mdp.robot_type_id)
        base_lin_vel = ObsTerm(func=cross_mdp.base_lin_vel_cross)
        base_ang_vel = ObsTerm(func=cross_mdp.base_ang_vel_cross, scale=0.2)
        projected_gravity = ObsTerm(func=cross_mdp.projected_gravity_cross)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=cross_mdp.joint_pos_cross)
        joint_vel = ObsTerm(func=cross_mdp.joint_vel_cross, scale=0.05)
        last_action = ObsTerm(func=cross_mdp.last_action_cross)

        def __post_init__(self):
            self.history_length = 1
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


##############################################################################
# Actions  (29 unified dims)
##############################################################################


@configclass
class CrossEmbodiedActionsCfg:
    """Unified joint-position actions."""

    joint_pos: CrossEmbodiedJointPosActionCfg = CrossEmbodiedJointPosActionCfg(
        scale_g1=0.5,
        scale_go2=0.25,
    )


##############################################################################
# Velocity commands  (shared, heading_command=False to avoid robot-asset dep.)
##############################################################################


@configclass
class CrossEmbodiedCommandsCfg:
    """Velocity commands.

    ``asset_name`` points to ``robot_g1``.  For Go2 envs the heading metric
    is computed from G1's parked pose (incorrect but harmless – it affects only
    TensorBoard logging, not reward computation).
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot_g1",
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
# Events (domain randomisation + resets)
##############################################################################


@configclass
class CrossEmbodiedEventsCfg:
    """Domain randomisation for the cross-embodied env.

    Reset events apply to BOTH robots; the active robot uses the randomised
    pose while the inactive robot is immediately re-parked by the env's
    ``_reset_idx`` override.
    """

    # G1 base reset
    reset_g1_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot_g1"),
        },
    )

    # G1 joints reset
    reset_g1_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_g1"),
        },
    )

    # Go2 base reset
    reset_go2_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot_go2"),
        },
    )

    # Go2 joints reset
    reset_go2_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_go2"),
        },
    )

    # Light mass randomisation for G1
    add_g1_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "asset_cfg": SceneEntityCfg("robot_g1", body_names="torso_link"),
        },
    )

    # Light mass randomisation for Go2
    add_go2_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "asset_cfg": SceneEntityCfg("robot_go2", body_names="base"),
        },
    )


##############################################################################
# Rewards
##############################################################################


@configclass
class CrossEmbodiedRewardsCfg:
    """Shared reward terms for both G1 and Go2 robots."""

    # --- task rewards ---
    track_lin_vel_xy = RewTerm(
        func=cross_mdp.track_lin_vel_xy_cross,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=cross_mdp.track_ang_vel_z_cross,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=cross_mdp.is_alive_cross, weight=0.15)

    # --- base penalties ---
    lin_vel_z_l2 = RewTerm(func=cross_mdp.lin_vel_z_l2_cross, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=cross_mdp.ang_vel_xy_l2_cross, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=cross_mdp.flat_orientation_l2_cross, weight=-5.0)

    # --- joint / action penalties ---
    joint_vel = RewTerm(func=cross_mdp.joint_vel_l2_cross, weight=-0.001)
    joint_acc = RewTerm(func=cross_mdp.joint_acc_l2_cross, weight=-2.5e-7)
    action_rate = RewTerm(func=cross_mdp.action_rate_l2_cross, weight=-0.05)
    dof_pos_limits = RewTerm(func=cross_mdp.joint_pos_limits_cross, weight=-5.0)

    # --- G1-only joint deviation penalties (zero for Go2 envs) ---
    # Prevent G1's arms, waist, and hip roll/yaw from drifting far from default.
    joint_deviation_g1_arms = RewTerm(
        func=cross_mdp.joint_deviation_g1_l1_cross,
        weight=-0.1,
        params={
            "joint_names": [
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*",
            ]
        },
    )
    joint_deviation_g1_waist = RewTerm(
        func=cross_mdp.joint_deviation_g1_l1_cross,
        weight=-1.0,
        params={"joint_names": ["waist.*"]},
    )
    joint_deviation_g1_legs = RewTerm(
        func=cross_mdp.joint_deviation_g1_l1_cross,
        weight=-1.0,
        params={"joint_names": [".*_hip_roll_joint", ".*_hip_yaw_joint"]},
    )


##############################################################################
# Terminations
##############################################################################


@configclass
class CrossEmbodiedTerminationsCfg:
    """Termination terms for the cross-embodied env."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_fall = DoneTerm(
        func=cross_mdp.base_below_threshold_cross,
        params={"min_height": 0.25},
    )
    # bad_orientation = DoneTerm(
    #     func=cross_mdp.bad_orientation_cross,
    #     params={"limit_angle": 0.8},
    # )


##############################################################################
# Curriculum (empty – no terrain curriculum for flat env)
##############################################################################


@configclass
class CrossEmbodiedCurriculumCfg:
    pass


##############################################################################
# Environment configuration
##############################################################################


@configclass
class CrossEmbodiedG1Go2FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cross-embodied G1 + Go2 flat locomotion env."""

    scene: CrossEmbodiedG1Go2SceneCfg = CrossEmbodiedG1Go2SceneCfg(
        num_envs=512,
        env_spacing=4.0,
        replicate_physics=False,  # Required for heterogeneous robots
    )
    observations: CrossEmbodiedObsCfg = CrossEmbodiedObsCfg()
    actions: CrossEmbodiedActionsCfg = CrossEmbodiedActionsCfg()
    commands: CrossEmbodiedCommandsCfg = CrossEmbodiedCommandsCfg()
    rewards: CrossEmbodiedRewardsCfg = CrossEmbodiedRewardsCfg()
    terminations: CrossEmbodiedTerminationsCfg = CrossEmbodiedTerminationsCfg()
    events: CrossEmbodiedEventsCfg = CrossEmbodiedEventsCfg()
    curriculum: CrossEmbodiedCurriculumCfg = CrossEmbodiedCurriculumCfg()

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


##############################################################################
# Custom environment class
##############################################################################

# Height to park inactive robots above ground (m).
_PARK_ALTITUDE: float = 100.0


class CrossEmbodiedG1Go2Env(ManagerBasedRLEnv):
    """Isaac Lab environment with G1 and Go2 co-existing in a single scene.

    The first ``num_envs // 2`` environment instances run G1; the second half
    run Go2.  The inactive robot in each environment is parked 100 m above its
    environment origin to keep it clear of the simulation, and re-parked on
    every episode reset.

    Extra attributes
    ~~~~~~~~~~~~~~~~
    robot_type_mask : torch.Tensor, shape (N,), dtype long
        ``0`` for G1 envs, ``1`` for Go2 envs.
    is_g1_env : torch.Tensor, shape (N,), dtype bool
        ``True`` for G1 envs.
    robot_type_onehot : torch.Tensor, shape (N, 2)
        ``[1, 0]`` for G1, ``[0, 1]`` for Go2.
    """

    cfg: CrossEmbodiedG1Go2FlatEnvCfg

    def __init__(self, cfg: CrossEmbodiedG1Go2FlatEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        # Pre-initialize robot type tensors BEFORE super().__init__() because
        # ObservationManager._prepare_terms() (called inside load_managers())
        # probes obs shapes by calling each obs function once, which requires
        # these tensors to already exist on the env object.
        n = cfg.scene.num_envs
        n_g1 = n // 2
        device = cfg.sim.device

        self.robot_type_mask = torch.zeros(n, dtype=torch.long, device=device)
        self.robot_type_mask[n_g1:] = 1
        self.is_g1_env: torch.Tensor = self.robot_type_mask == 0  # (N,) bool
        self.robot_type_onehot = torch.zeros(n, 2, device=device)
        self.robot_type_onehot[:n_g1, 0] = 1.0   # G1 → [1, 0]
        self.robot_type_onehot[n_g1:, 1] = 1.0   # Go2 → [0, 1]

        # gymnasium passes all registered kwargs (e.g. env_cfg_entry_point,
        # rsl_rl_cfg_entry_point) into the constructor; discard them here.
        super().__init__(cfg, render_mode)

        # --- Initial parking ------------------------------------------------
        all_g1_ids = self.is_g1_env.nonzero(as_tuple=False).view(-1)
        all_go2_ids = (~self.is_g1_env).nonzero(as_tuple=False).view(-1)
        self._park_robot("robot_go2", all_g1_ids)   # park Go2 in G1 envs
        self._park_robot("robot_g1", all_go2_ids)   # park G1 in Go2 envs

    # ------------------------------------------------------------------
    # Parking helpers
    # ------------------------------------------------------------------

    def _park_robot(self, robot_name: str, env_ids: torch.Tensor) -> None:
        """Teleport *robot_name* to ``_PARK_ALTITUDE`` m above each env origin."""
        if len(env_ids) == 0:
            return
        robot = self.scene[robot_name]
        # World-frame position: env origin XY + 100 m altitude
        origins = self.scene.env_origins[env_ids]           # (M, 3)
        pos = origins.clone()
        pos[:, 2] = _PARK_ALTITUDE

        # Identity quaternion  [w, x, y, z]
        quat = torch.zeros(len(env_ids), 4, device=self.device)
        quat[:, 0] = 1.0

        # Zero velocity
        vel = torch.zeros(len(env_ids), 6, device=self.device)

        robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        robot.write_root_velocity_to_sim(vel, env_ids=env_ids)

        # Hold joints at default
        robot.set_joint_position_target(
            robot.data.default_joint_pos[env_ids],
            env_ids=env_ids,
        )
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[env_ids],
            torch.zeros_like(robot.data.default_joint_pos[env_ids]),
            env_ids=env_ids,
        )

    # ------------------------------------------------------------------
    # Per-step re-parking – prevents parked robots from gravity drift
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        """Re-park inactive robots before each env step to prevent gravity drift."""
        all_ids = torch.arange(self.num_envs, device=self.device)
        g1_ids = all_ids[self.is_g1_env]
        go2_ids = all_ids[~self.is_g1_env]
        if len(g1_ids) > 0:
            self._park_robot("robot_go2", g1_ids)
        if len(go2_ids) > 0:
            self._park_robot("robot_g1", go2_ids)
        return super().step(action)

    # ------------------------------------------------------------------
    # Reset override – re-park inactive robots after standard reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor) -> None:  # type: ignore[override]
        """Standard env reset + re-park of inactive robots."""
        super()._reset_idx(env_ids)

        # Determine which sub-set of reset envs belong to each robot type.
        g1_ids = env_ids[self.is_g1_env[env_ids]]
        go2_ids = env_ids[~self.is_g1_env[env_ids]]

        # Re-park the INACTIVE robot so it doesn't drift after physics reset.
        if len(g1_ids) > 0:
            self._park_robot("robot_go2", g1_ids)
        if len(go2_ids) > 0:
            self._park_robot("robot_g1", go2_ids)
