# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
import legged_rl_lab.tasks.locomotion.velocity.mdp as mdp
from legged_rl_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        height_scanner = ObsTerm(func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = False
            self.concatenate_terms = True
            

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class G1RewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # -- base
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0) #
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
       
    # -- joints
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
        )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )#

    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.5,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward_humanoid,
        weight=2.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "foot_scanner_cfgs": [
                SceneEntityCfg("foot_scanner_l"),
                SceneEntityCfg("foot_scanner_r"),
            ],
        },
    )#

    
    # gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.8,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #     },
    # )

    # -- posture
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_knee_joint"])},
    )#
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )#
    

    
@configclass
class G1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )


@configclass
class UnitreeG1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: G1ObservationsCfg = G1ObservationsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()   
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        base_link_name = "torso_link"
        foot_link_name = ".*_ankle_roll_link"

        # ------------------------------ Scene ------------------------------
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # 每只脚一个向下单射线 RayCaster，用于奖励函数中获取脚下地形高度
        # offset.pos Z=0.5 保证射线从脚踝以上发出，不会因脚踝本身几何体而提前终止
        _foot_scanner_cfg = RayCasterCfg(
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            ray_alignment="world",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.0, 0.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.foot_scanner_l = _foot_scanner_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link"
        )
        self.scene.foot_scanner_r = _foot_scanner_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link"
        )
        self.scene.foot_scanner_l.update_period = self.decimation * self.sim.dt
        self.scene.foot_scanner_r.update_period = self.decimation * self.sim.dt
        
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        
        # ------------------------------ Events ------------------------------
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = base_link_name
        self.events.reset_base.params = {
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
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_com = None

        # ------------------------------ Actions ------------------------------
        self.actions.joint_pos.scale = 0.25
        # self.actions.joint_pos.clip = {".*": (-1.0, 1.0)}

        # ------------------------------ Rewards ------------------------------
        self.rewards.undesired_contacts = None
        # ------------------------------ Curriculum ------------------------------

        
        # ------------------------------- Commands ------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.debug_vis = False

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeG1RoughEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeG1RoughEnvCfg_PLAY(UnitreeG1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 初始 spawn 在 level 0（平面列），让机器人从平面出发
        self.scene.terrain.max_init_terrain_level = 0

        # -----------------------------------------------------------------------
        # Play 专用地形：4种障碍紧邻排列（1行×4列），周边20m平地边框
        #
        # TerrainGenerator 的布局规则：
        #   - num_cols 决定地形类型数（每列1种），num_rows 决定难度行数
        #   - 2×2 布局每列只能分配2种，因此 4种各1个必须用 num_cols=4, num_rows=1
        #
        #   列0: 凹凸不平 (random_rough)
        #   列1: 金字塔楼梯 (pyramid_stairs)
        #   列2: 倒金字塔楼梯 (pyramid_stairs_inv)
        #   列3: 金字塔坡面 (hf_pyramid_slope)
        #
        # border_width=20m：周边是宽阔平地，机器人可以在上面行走
        # 子地形 border_width=0：相邻地形零间距接壤
        # -----------------------------------------------------------------------
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            seed=42,
            size=(12.0, 12.0),       # 每块地形 12×12m，正方形
            border_width=20.0,        # 周边 20m 平地
            border_height=0.0,        # 平地边框与地面齐平
            num_rows=1,               # 1行：只有1个难度
            num_cols=4,               # 4列：4种地形各1个
            curriculum=True,          # 严格按比例分配，确保每列对应一种
            difficulty_range=(0.5, 0.5),  # level 6 难度（训练10行，6/10=0.6）
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            sub_terrains={
                # 列0: 凹凸不平（border_width=0 与相邻地形零间距接壤）
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(0.02, 0.10),
                    noise_step=0.01,
                    border_width=0.0,
                ),
                # 列1: 金字塔楼梯（向上）
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=1.0,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=0.0,
                    holes=False,
                ),
                # 列2: 倒金字塔楼梯（向下）
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=1.0,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=0.0,
                    holes=False,
                ),
                # 列3: 金字塔坡面
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=1.0,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.0,
                    inverted=False,
                ),
            },
        )

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None