from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_rl_lab.tasks.parkour.attention.attention_env_cfg import (
    AttentionCriticTerrainMapCfg,
    AttentionEnvCfgMixin,
    AttentionObservationsCfg,
    AttentionPolicyCfg,
    AttentionTerrainMapCfg,
    attention_height_scanner_cfg,
)
from legged_rl_lab.tasks.parkour.depth.config.go2.go2_depth_env_cfg import (
    Go2TSDepthEnvCfg,
    Go2TSDepthEventCfg,
    Go2TSDepthRewardsCfg,
    Go2TSDepthSceneCfg,
    Go2TSDepthTerminationsCfg,
)
import legged_rl_lab.tasks.parkour.attention.mdp as mdp


_GO2_CONTACT_LINKS: tuple[str, ...] = (".*_thigh", ".*_calf", ".*_foot")
_GO2_FOOT_SENSORS: tuple[str, ...] = (
    "foot_height_scanner_FL",
    "foot_height_scanner_FR",
    "foot_height_scanner_RL",
    "foot_height_scanner_RR",
)
_GO2_FOOT_BODIES: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
_GO2_HEIGHT_SCANNER_PRIM_PATH = "{ENV_REGEX_NS}/Robot/base"


@configclass
class Go2AttentionSceneCfg(Go2TSDepthSceneCfg):
    depth_scanner = None
    height_scanner = attention_height_scanner_cfg(_GO2_HEIGHT_SCANNER_PRIM_PATH)


@configclass
class Go2AttentionCriticCfg(ObsGroup):
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        scale=(1.0, 1.0, 0.25),
    )
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    actions = ObsTerm(func=mdp.last_action, scale=0.1)
    dr_friction = ObsTerm(func=mdp.scalar_rigid_friction_mean, params={"asset_cfg": SceneEntityCfg("robot")})
    dr_mass_scale = ObsTerm(
        func=mdp.body_mass_scale,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    dr_com_b = ObsTerm(
        func=mdp.body_com_pos_b,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    dr_push_xy = ObsTerm(func=mdp.last_push_delta_xy)
    dr_kp_scale = ObsTerm(
        func=mdp.joint_stiffness_scale,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dr_kd_scale = ObsTerm(
        func=mdp.joint_damping_scale,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    link_contact_states = ObsTerm(
        func=mdp.links_contact_binary,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(_GO2_CONTACT_LINKS)),
            "threshold": 1.0,
        },
    )
    height_relative_to_feet = ObsTerm(
        func=mdp.height_relative_to_feet,
        params={
            "sensor_names": list(_GO2_FOOT_SENSORS),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(_GO2_FOOT_BODIES)),
            "clip": (-1.0, 1.0),
        },
    )
    normal_vector_around_feet = ObsTerm(
        func=mdp.normal_vector_around_feet,
        params={"sensor_names": list(_GO2_FOOT_SENSORS)},
    )

    def __post_init__(self):
        self.history_length = 5
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2AttentionObservationsCfg(AttentionObservationsCfg):
    policy: AttentionPolicyCfg = AttentionPolicyCfg()
    critic: Go2AttentionCriticCfg = Go2AttentionCriticCfg()
    terrain_map: AttentionTerrainMapCfg = AttentionTerrainMapCfg()
    critic_terrain_map: AttentionCriticTerrainMapCfg = AttentionCriticTerrainMapCfg()


@configclass
class Go2AttentionEnvCfg(AttentionEnvCfgMixin, Go2TSDepthEnvCfg):
    scene: Go2AttentionSceneCfg = Go2AttentionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: Go2AttentionObservationsCfg = Go2AttentionObservationsCfg()
    rewards: Go2TSDepthRewardsCfg = Go2TSDepthRewardsCfg()
    events: Go2TSDepthEventCfg = Go2TSDepthEventCfg()
    terminations: Go2TSDepthTerminationsCfg = Go2TSDepthTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.configure_attention_train(_GO2_HEIGHT_SCANNER_PRIM_PATH)


@configclass
class Go2AttentionEnvCfg_PLAY(Go2AttentionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.configure_attention_play()
