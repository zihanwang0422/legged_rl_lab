from __future__ import annotations

import copy

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from legged_rl_lab.terrains import AME_PARKOUR_TERRAINS_CFG, HfAlternateColumnStakesTerrainCfg
import legged_rl_lab.tasks.parkour.attention.mdp as mdp


ATTENTION_GRID_RESOLUTION = 0.1
ATTENTION_GRID_SIZE = (1.6, 1.0)
ATTENTION_MAP_SCAN_DIM = (17, 11, 3)
ATTENTION_OBS_GROUPS = {
    "actor": ["policy", "terrain_map"],
    "critic": ["critic", "critic_terrain_map"],
}


def attention_height_scanner_cfg(prim_path: str) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=ATTENTION_GRID_RESOLUTION, size=ATTENTION_GRID_SIZE),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class AttentionPolicyCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        scale=(1.0, 1.0, 0.25),
    )
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05)
    actions = ObsTerm(func=mdp.last_action, scale=0.1)

    def __post_init__(self):
        self.history_length = 1
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class AttentionTerrainMapCfg(ObsGroup):
    terrain_map = ObsTerm(
        func=mdp.elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": True},
    )

    def __post_init__(self):
        self.history_length = 1
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class AttentionCriticTerrainMapCfg(ObsGroup):
    terrain_map = ObsTerm(
        func=mdp.elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
    )

    def __post_init__(self):
        self.history_length = 1
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class AttentionObservationsCfg:
    policy: AttentionPolicyCfg = AttentionPolicyCfg()
    terrain_map: AttentionTerrainMapCfg = AttentionTerrainMapCfg()
    critic_terrain_map: AttentionCriticTerrainMapCfg = AttentionCriticTerrainMapCfg()


class AttentionEnvCfgMixin:
    def configure_attention_train(self, height_scanner_prim_path: str) -> None:
        self.scene.terrain.terrain_generator = copy.deepcopy(AME_PARKOUR_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = 4

        self.scene.height_scanner.prim_path = height_scanner_prim_path
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=ATTENTION_GRID_RESOLUTION,
            size=ATTENTION_GRID_SIZE,
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

    def configure_attention_play(self) -> None:
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)

        self.observations.policy.enable_corruption = False
        self.observations.terrain_map.enable_corruption = False
        self.observations.terrain_map.terrain_map.params["noise"] = False
        self.scene.height_scanner.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.size = (8.0, 8.0)
            self.scene.terrain.terrain_generator.sub_terrains = {
                "stakes": HfAlternateColumnStakesTerrainCfg(
                    proportion=0.5,
                    stake_height_max=0.0,
                    stake_side_range=(0.2, 0.2),
                    stake_gap_range=(0.3, 0.3),
                    column_gap_range=(0.3, 0.3),
                    column_jitter=0.0,
                    holes_depth=-2.0,
                    platform_width=2.0,
                    border_width=0.25,
                ),
            }

        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
        if hasattr(self.events, "actuator_gains"):
            self.events.actuator_gains = None
