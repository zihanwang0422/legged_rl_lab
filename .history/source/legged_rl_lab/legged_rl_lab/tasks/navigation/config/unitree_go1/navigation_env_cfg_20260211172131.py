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
# Scene Configuration for Go1 Navigation
##

@configclass
class Go1NavigationSceneCfg(NavigationSceneCfg):
    """Scene configuration for Go1 navigation."""

    # Override robot
    robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Update height_scanner prim_path for Go1's base link
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",  # Go1's base link
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.3]),  # 4x3 = 12 rays
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


##
# Low-Level Observations Configuration for Go1
##

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

@configclass  
class Go1LowLevelObservationsCfg:
    """Low-level observations for Go1 locomotion policy.
    
    This matches the observation space used during Go1 flat locomotion training (48 dims).
    The trained policy does NOT use height_scan.
    
    Observation breakdown:
    - base_lin_vel: 3 dims
    - base_ang_vel: 3 dims  
    - projected_gravity: 3 dims
    - velocity_commands: 3 dims (vx, vy, omega)
    - joint_pos_rel: 12 dims (12 joints)
    - joint_vel: 12 dims (12 joints)
    - actions: 12 dims (last actions)
    Total: 48 dims
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the low-level locomotion policy."""

        # Copy from Go1 flat locomotion observations (NO height_scan)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=0.2, clip=(-100.0, 100.0), noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100.0, 100.0), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=1.0,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, scale=1.0, clip=(-100.0, 100.0), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0), noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action, scale=1.0, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Actions Configuration for Go1
##

@configclass
class Go1ActionsCfg(ActionsCfg):
    """Action configuration for Go1 using pre-trained locomotion policy."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=GO1_FLAT_POLICY_PATH,
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=Go1LowLevelObservationsCfg().policy,  # Use custom observations with height_scan
    )


##
# Environment Configuration
##

@configclass
class UnitreeGo1NavigationEnvCfg(NavigationEnvCfg):
    """Configuration for the Go1 navigation environment on flat terrain."""

    # Override scene and actions with Go1-specific configs
    scene: Go1NavigationSceneCfg = Go1NavigationSceneCfg(num_envs=2048, env_spacing=2.5)
    actions: Go1ActionsCfg = Go1ActionsCfg()

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

        # Update termination body name to Go1's base
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"


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
class ObstacleNavigationSceneCfg(Go1NavigationSceneCfg):
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

    # Use larger height scan area for better obstacle detection
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 16x10 = 160 rays for better perception
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class UnitreeGo1NavigationObstacleEnvCfg(UnitreeGo1NavigationEnvCfg):
    """Configuration for the Go1 navigation environment with cylinder obstacles."""

    # Override scene with obstacle terrain
    scene: ObstacleNavigationSceneCfg = ObstacleNavigationSceneCfg(num_envs=2048, env_spacing=2.5)
    
    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # Slightly longer episode time for obstacle navigation
        self.commands.pose_command.resampling_time_range = (12.0, 12.0)
        self.episode_length_s = 12.0


@configclass
class UnitreeGo1NavigationObstacleEnvCfg_PLAY(UnitreeGo1NavigationObstacleEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
