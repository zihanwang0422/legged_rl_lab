# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshRepeatedCylindersTerrainCfg, MeshPlaneTerrainCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import legged_rl_lab.tasks.navigation.mdp as mdp

# Import base navigation environment configuration
from legged_rl_lab.tasks.navigation.navigation_env_vfg import (
    NavigationEnvCfg,
    NavigationSceneCfg,
    ObservationsCfg,
    ActionsCfg,
    CommandsCfg,
)

# Import Go1 flat locomotion config as the low-level policy base
from legged_rl_lab.tasks.locomotion.velocity.config.quadruped.unitree_go1.flat_env_cfg import UnitreeGo1FlatEnvCfg

##
# Pre-defined configs
##
from legged_rl_lab.assets.unitree import UNITREE_GO1_CFG  # isort: skip

# Instantiate the low-level env config for Go1
LOW_LEVEL_ENV_CFG = UnitreeGo1FlatEnvCfg()

# Path to pre-trained Go1 flat locomotion policy
# NOTE: Update this path to your best trained Go1 flat locomotion policy
_LEGGED_RL_LAB_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
))))
GO1_FLAT_POLICY_PATH = os.path.join(
    _LEGGED_RL_LAB_DIR, "logs", "rsl_rl", "unitree_go1_flat", "2026-02-01_18-43-20", "exported", "policy.pt"
)


##
# Obstacle Terrain Configuration - Cylinder obstacles for navigation
##

NAVIGATION_OBSTACLE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    border_height=0.5,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
        "cylinders_sparse": MeshRepeatedCylindersTerrainCfg(
            proportion=0.3,
            platform_width=2.0,
            object_params_start=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=5,
                height=0.5,
                radius=0.1,
            ),
            object_params_end=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=8,
                height=0.8,
                radius=0.15,
            ),
        ),
        "cylinders_medium": MeshRepeatedCylindersTerrainCfg(
            proportion=0.3,
            platform_width=1.5,
            object_params_start=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=10,
                height=0.5,
                radius=0.08,
            ),
            object_params_end=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=15,
                height=1.0,
                radius=0.12,
            ),
        ),
        "cylinders_dense": MeshRepeatedCylindersTerrainCfg(
            proportion=0.2,
            platform_width=1.0,
            object_params_start=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=15,
                height=0.5,
                radius=0.06,
            ),
            object_params_end=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=20,
                height=1.0,
                radius=0.1,
            ),
        ),
    },
)


##
# Scene Configuration for Navigation
##

@configclass
class NavigationSceneCfg(InteractiveSceneCfg):
    """Configuration for the navigation scene with Go1 robot."""

    # ground terrain - flat by default, overridden for obstacle env
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
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
    # robot
    robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = None  # no height scanner for flat navigation
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##

@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP.
    
    Uses a pre-trained Go1 locomotion policy as the low-level controller.
    The high-level navigation policy outputs velocity commands (vx, vy, omega)
    which are fed to the low-level policy.
    """

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        # Path to the exported Go1 flat locomotion policy
        policy_path=GO1_FLAT_POLICY_PATH,
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the navigation policy.
        
        The navigation policy observes:
        - Base linear velocity (3D)
        - Projected gravity (3D) 
        - Pose command to target (4D: x, y, z, heading)
        """

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Termination penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    
    # Position tracking - coarse: encourage moving toward goal
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    # Position tracking - fine: encourage precise arrival at goal
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    # Heading tracking - encourage correct orientation at goal
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP.
    
    Generates random target poses (x, y, heading) for the robot to navigate to.
    """

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )


##
# Environment Configuration
##

@configclass
class UnitreeGo1NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go1 navigation environment on flat terrain."""

    # environment settings
    scene: NavigationSceneCfg = NavigationSceneCfg(num_envs=2048, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # Use the same dt as Go1 flat locomotion
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt  # 0.005
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation  # 4
        # High-level decimation: low_level_decimation * high_level_factor
        # Low-level runs at 4 * 0.005 = 0.02s, high-level at 0.02 * 10 = 0.2s
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10  # 40
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        # Update sensor periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UnitreeGo1NavigationEnvCfg_PLAY(UnitreeGo1NavigationEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


##
# Obstacle Navigation Environment Configuration
##

@configclass
class ObstacleNavigationSceneCfg(NavigationSceneCfg):
    """Scene configuration with cylinder obstacle terrain for navigation."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=NAVIGATION_OBSTACLE_TERRAINS_CFG,
        max_init_terrain_level=None,
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

    # Add height scanner for obstacle avoidance observation
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ObstacleObservationsCfg:
    """Observation specifications for the obstacle navigation MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the obstacle navigation policy.

        The navigation policy observes:
        - Base linear velocity (3D)
        - Projected gravity (3D)
        - Pose command to target (4D: x, y, z, heading)
        - Height scan for obstacle detection
        """

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            scale=1.0,
            clip=(-1.0, 1.0),
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass 
class ObstacleRewardsCfg:
    """Reward terms for the obstacle navigation MDP."""

    # Termination penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)

    # Position tracking - coarse: encourage moving toward goal
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    # Position tracking - fine: encourage precise arrival at goal
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    # Heading tracking - encourage correct orientation at goal
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )


@configclass
class ObstacleCommandsCfg:
    """Command terms for obstacle navigation.
    
    Slightly reduced range compared to flat terrain due to obstacles.
    """

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(12.0, 12.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class UnitreeGo1NavigationObstacleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go1 navigation environment with cylinder obstacles."""

    # environment settings
    scene: ObstacleNavigationSceneCfg = ObstacleNavigationSceneCfg(num_envs=2048, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObstacleObservationsCfg = ObstacleObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: ObstacleCommandsCfg = ObstacleCommandsCfg()
    rewards: ObstacleRewardsCfg = ObstacleRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt  # 0.005
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation  # 4
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10  # 40
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        # Update sensor periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UnitreeGo1NavigationObstacleEnvCfg_PLAY(UnitreeGo1NavigationObstacleEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
