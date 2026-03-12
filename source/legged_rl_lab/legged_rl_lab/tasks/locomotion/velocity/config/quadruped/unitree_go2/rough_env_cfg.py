# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
import isaaclab.terrains as terrain_gen

@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    foot_link_name = ".*_foot" 
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # observations
        self.observations.policy.history_length = 5
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.critic.history_length = 5
        
        # action
        self.actions.joint_pos.scale = 0.25

        #------------------------------- Event -------------------------------
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
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
        self.events.base_com = None

        # ==================== Rewards Configuration ====================
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.joint_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # # ===== General Rewards =====
        # self.rewards.is_alive = None
        # self.rewards.is_terminated = None
        
        # # ===== Base Rewards =====
        # # Tracking rewards
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.75
        
        # # Base penalties
        # self.rewards.lin_vel_z_l2.weight = -2.0
        # self.rewards.ang_vel_xy_l2.weight = -0.05  
        # self.rewards.flat_orientation_l2.weight = 0.0
        # self.rewards.base_height_l2.weight = 0.0
        
        # self.rewards.body_lin_acc_l2 = None
        
        # # ===== Joint Rewards =====
        # self.rewards.joint_torques_l2.weight = -2e-4
        # self.rewards.joint_vel_l1 = None
        # self.rewards.joint_vel_l2.weight = -0.001  
        # self.rewards.joint_acc_l2.weight = -2.0e-7  
        # self.rewards.joint_deviation_l1 = None
        # self.rewards.joint_pos_limits.weight = -10.0
        # self.rewards.joint_vel_limits = None
        # self.rewards.applied_torque_limits = None
        # self.rewards.joint_power = None
        
        # # Action penalties 
        # self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.action_l2 = None
        
        # # ===== Contact Rewards =====
        # # Undesired contacts 
        # self.rewards.undesired_contacts.weight = -1.0
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]  
        # self.rewards.contact_forces = None
        
        # # Feet rewards - 关键：鼓励正常步态
        # self.rewards.feet_air_time.weight = 0.1  # 大幅增加权重，鼓励足够的腾空时间
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.params["threshold"] = 0.5  # 设置最小腾空时间阈值
        # self.rewards.feet_height = None
        # self.rewards.feet_slide.weight = -0.1  # 启用，惩罚脚部滑动
        
        # # ===== Other/Custom Rewards =====
        # # Standing still
        # self.rewards.stand_till.weight = -0.5  # 减小惩罚，避免在低速时过度惩罚
        # self.rewards.stand_till.params["command_name"] = "base_velocity"
        
        # # Body orientation
        # self.rewards.body_roll_l2.weight = -2.0
        # self.rewards.body_pitch_l2.weight = -5.0
        
        # # Diagonal gait symmetry (Trot Gait: FL+RR, FR+RL)
        # self.rewards.joint_symmetry_l2.weight = -0.1
        # self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
        #     ["FL_.*_joint", "RR_.*_joint"],  # 前左 + 后右 (对角线1)
        #     ["FR_.*_joint", "RL_.*_joint"],  # 前右 + 后左 (对角线2)
        # ]
        # self.rewards.action_symmetry_l2.weight = -0.02  # 降低权重，防止数值爆炸
        # self.rewards.action_symmetry_l2.params["mirror_joints"] = [
        #     ["FL_.*_joint", "RR_.*_joint"],  # 前左 + 后右 (对角线1)
        #     ["FR_.*_joint", "RL_.*_joint"],  # 前右 + 后左 (对角线2)
        # ]
        
        # # ===== Handstand/Footstand Rewards =====
        # self.rewards.handstand_feet_height_exp = None
        # self.rewards.handstand_feet_on_air = None
        # self.rewards.handstand_feet_air_time = None
        # self.rewards.handstand_orientation_l2 = None

        # terminations
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
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