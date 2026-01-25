# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from legged_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import legged_rl_lab.tasks.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from legged_rl_lab.assets.unitree import UNITREE_GO1_CFG  # isort: skip


@configclass
class UnitreeGo1FootstandEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        # ====Terrain Cfg====
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # action
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = "base"
        # apply to all rigid bodies except the base
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [r"^(?!.*base).*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = "base"
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        
        # ==================== Rewards Configuration ====================
        
        # ===== General Rewards =====
        self.rewards.is_terminated.weight = 0.0
        
        # ===== Base Rewards =====
        # Tracking 
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        
        # Base penalties
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = "base"
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = "base"

        
        # ===== Joint Rewards =====
        self.rewards.joint_torques_l2.weight = -1e-3
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -2.5e-6
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = -2e-4
        
        # ===== Contact Rewards =====
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_thigh"]
        self.rewards.contact_forces.weight = 0.0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = ".*_foot"
        
   
        
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_foot"
        
        # ===== Other/Custom Rewards =====
        self.rewards.stand_till.weight = 0.0
        
        # ===== Handstand/Footstand Rewards =====
        # 1. Height Reward
        hand_stand_type = "rear" # which leg on the ground "rear" or "front" or "hip" or "left" or "right"
        if hand_stand_type == "rear":
            air_leg_name = "F.*_foot"  
            feet_height_weight = 10.0
            feet_height = 0.6
            feet_on_air_weight = 10.0
            feet_air_time_weight = 5.0
            target_gravity_weight = -2.5
            target_gravity = [-1.0, 0.0, 0.0]
        
        elif hand_stand_type == "front":
            self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = "R.*_foot"
        
        elif hand_stand_type == "hip":   
            air_leg_name = "F.*_foot"  
            feet_height_weight = 10.0
            feet_height = 0.6
            feet_on_air_weight = 10.0
            feet_air_time_weight = 5.0
            target_gravity_weight = -2.0
            target_gravity = [-1.0, 0.0, 0.0]
        
        elif hand_stand_type == "left":         
            self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = "L.*_foot" 
        
        elif hand_stand_type == "right":        
            self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = "R.*_foot"
        
        else:
            raise ValueError("Invalid hand_stand_type. Choose from 'rear', 'front', 'hip', 'left', 'right'.")   
        
        self.rewards.handstand_feet_height_exp.weight = feet_height_weight
        self.rewards.handstand_feet_height_exp.params["target_height"] = feet_height
        self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names =  air_leg_name

        # 2. Feet on Air Reward
        self.rewards.handstand_feet_on_air.weight = feet_on_air_weight
        self.rewards.handstand_feet_on_air.params["sensor_cfg"].body_names = air_leg_name

        # 3. Air Time Reward
        self.rewards.handstand_feet_air_time.weight = feet_air_time_weight
        self.rewards.handstand_feet_air_time.params["sensor_cfg"].body_names = air_leg_name
        # 4. Orientation Reward
        self.rewards.handstand_orientation_l2.weight = target_gravity_weight
        self.rewards.handstand_orientation_l2.params["target_gravity"] = target_gravity  

        # 5. Contact Rewards - Critical for stability!
        # Reward rear feet staying on ground (positive reward for contact)
        # self.rewards.desired_rear_feet_contact = RewTerm(
        #     func=mdp.desired_contacts,
        #     weight=rear_feet_contact_weight,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"), "threshold": 1.0},
        # )
        # # Penalize front feet touching ground
        # self.rewards.undesired_front_feet_contact = RewTerm(
        #     func=mdp.undesired_contacts,
        #     weight=front_feet_penalty_weight,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"), "threshold": 1.0},
        # )

        # # 6. Symmetry Rewards - ensure left-right symmetry for both front and rear legs
        # self.rewards.joint_symmetry_l2.weight = joint_symmetry_weight
        # self.rewards.joint_symmetry_l2.params["mirror_joints"] = [
        #     ["RL_.*_joint", "RR_.*_joint"], 
        # ]
        
        # self.rewards.action_symmetry_l2.weight = action_symmetry_weight
        # self.rewards.action_symmetry_l2.params["mirror_joints"] = [
        #     ["RL_.*_joint", "RR_.*_joint"], 
        # ]
        
        # terminations
        # terminate on contact for any body except feet
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["(?!.*_foot).*"]
        
        #------------------------------- Commands -------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)  
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0) 
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        # Disable all rewards with zero weight
        self.disable_zero_weight_rewards() 


@configclass
class UnitreeGo1FootstandEnvCfg_PLAY(UnitreeGo1FootstandEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
  
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None