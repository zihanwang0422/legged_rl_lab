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
from legged_rl_lab.tasks.parkour.depth.config.g1.g1_depth_env_cfg import (
    G1TSDepthEnvCfg,
    G1TSDepthEventCfg,
    G1TSDepthRewardsCfg,
    G1TSDepthSceneCfg,
    G1TSDepthTerminationsCfg,
)
import legged_rl_lab.tasks.parkour.attention.mdp as mdp


_G1_CONTACT_LINKS: tuple[str, ...] = (
    "torso_link",
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_ankle_pitch_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_ankle_pitch_link",
    "waist_yaw_link",
)
_G1_FOOT_SENSORS: tuple[str, ...] = ("foot_height_scanner_L", "foot_height_scanner_R")
_G1_FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
_G1_HEIGHT_SCANNER_PRIM_PATH = "{ENV_REGEX_NS}/Robot/torso_link"


@configclass
class G1AttentionSceneCfg(G1TSDepthSceneCfg):
    depth_scanner = None
    height_scanner = attention_height_scanner_cfg(_G1_HEIGHT_SCANNER_PRIM_PATH)


@configclass
class G1AttentionCriticCfg(ObsGroup):
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
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    dr_com_b = ObsTerm(
        func=mdp.body_com_pos_b,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
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
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(_G1_CONTACT_LINKS)),
            "threshold": 1.0,
        },
    )
    height_relative_to_feet = ObsTerm(
        func=mdp.height_relative_to_feet,
        params={
            "sensor_names": list(_G1_FOOT_SENSORS),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(_G1_FOOT_BODIES)),
            "clip": (-1.0, 1.0),
        },
    )
    normal_vector_around_feet = ObsTerm(
        func=mdp.normal_vector_around_feet,
        params={"sensor_names": list(_G1_FOOT_SENSORS)},
    )
    gait_phase = ObsTerm(func=mdp.gait_phase_sin_cos, params={"period": 0.8, "offset": [0.0, 0.5]})
    feet_pos = ObsTerm(
        func=mdp.feet_pos_body_frame,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=list(_G1_FOOT_BODIES))},
    )
    feet_vel = ObsTerm(
        func=mdp.feet_vel_body_frame,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=list(_G1_FOOT_BODIES))},
    )
    feet_force = ObsTerm(
        func=mdp.feet_contact_force,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    root_height = ObsTerm(func=mdp.base_pos_z)

    def __post_init__(self):
        self.history_length = 5
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1AttentionObservationsCfg(AttentionObservationsCfg):
    policy: AttentionPolicyCfg = AttentionPolicyCfg()
    critic: G1AttentionCriticCfg = G1AttentionCriticCfg()
    terrain_map: AttentionTerrainMapCfg = AttentionTerrainMapCfg()
    critic_terrain_map: AttentionCriticTerrainMapCfg = AttentionCriticTerrainMapCfg()


@configclass
class G1AttentionEnvCfg(AttentionEnvCfgMixin, G1TSDepthEnvCfg):
    scene: G1AttentionSceneCfg = G1AttentionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1AttentionObservationsCfg = G1AttentionObservationsCfg()
    rewards: G1TSDepthRewardsCfg = G1TSDepthRewardsCfg()
    events: G1TSDepthEventCfg = G1TSDepthEventCfg()
    terminations: G1TSDepthTerminationsCfg = G1TSDepthTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.configure_attention_train(_G1_HEIGHT_SCANNER_PRIM_PATH)


@configclass
class G1AttentionEnvCfg_PLAY(G1AttentionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.configure_attention_play()
