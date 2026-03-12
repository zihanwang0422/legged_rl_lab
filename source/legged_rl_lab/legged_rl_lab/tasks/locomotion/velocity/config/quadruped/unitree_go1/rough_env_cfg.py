# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from legged_rl_lab.assets.unitree import UNITREE_GO1_CFG  # isort: skip
import isaaclab.terrains as terrain_gen

@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        
        # #------------------------------- Terrain -------------------------------
        # self.scene.terrain.terrain_generator.num_rows = 3  # 默认 10
        # self.scene.terrain.terrain_generator.num_cols = 3  # 默认 10
        # self.scene.terrain.terrain_generator.horizontal_scale = 0.2  # 增大以减少顶点数
        # self.scene.terrain.terrain_generator.vertical_scale = 0.005
        # self.scene.terrain.terrain_generator.sub_terrains = {
        #     "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #         proportion=0.25,
        #         step_height_range=(0.05, 0.23),
        #         step_width=0.3,
        #         platform_width=3.0,
        #         border_width=1.0,
        #         holes=False,
        #     ),
        #     "inverted_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #         proportion=0.25,
        #         step_height_range=(0.05, 0.23),
        #         step_width=0.3,
        #         platform_width=3.0,
        #         border_width=1.0,
        #         holes=False,
        #     ),
        #     "pyramid_slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
        #         proportion=0.25,
        #         slope_range=(0.0, 0.4),
        #         platform_width=3.0,
        #         border_width=0.25,
        #         inverted=False,
        #     ),
        #     "inverted_slopes": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #         proportion=0.25,
        #         slope_range=(0.0, 0.4),
        #         platform_width=3.0,
        #         border_width=0.25,
        #     ),
        # }

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
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
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


        # rewards
        self.rewards.undesired_contacts = None #
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_(trunk|calf|thigh)"
        
        self.rewards.joint_torques_l2.weight = -0.0002
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.005
        
        # self.rewards.base_height_l2.weight = -2.04
        # self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
 

        
        
        #------------------------------- Terminations -------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = "trunk"
        
        
        #------------------------------- Commands -------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0) 
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)  
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8) 



@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
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
