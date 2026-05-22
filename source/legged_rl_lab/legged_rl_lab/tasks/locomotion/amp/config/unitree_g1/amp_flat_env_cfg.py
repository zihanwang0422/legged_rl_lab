# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP environment configurations for Unitree G1 humanoid robot."""

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_rl_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAMPRoughEnvCfg
from legged_rl_lab import LEGGED_RL_LAB_ROOT_DIR
import legged_rl_lab.tasks.locomotion.amp.mdp as mdp

##
# Pre-defined configs
##
from legged_rl_lab.assets.unitree import UNITREE_G1_29DOF_CFG  # isort: skip

@configclass
class UnitreeG1AMPFlatEnvCfg(LocomotionAMPRoughEnvCfg):
    """Unitree G1 humanoid AMP environment on flat terrain."""

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"

    # Key bodies for AMP / critic obs.  Order matters — must match the
    # body indices in motion_loader's G1 profile and the reference AMP design.
    key_body_names = (
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_rubber_hand",
        "right_rubber_hand",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ----------------------------- Control / Physics Rate -----------------------------
        self.sim.dt = 1.0 / 150.0
        self.decimation = 5
        self.sim.render_interval = self.decimation

        # ----------------------------- Scene -----------------------------
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No height scanner
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None

        # No terrain curriculum
        self.curriculum.terrain_levels = None

        # ----------------------------- Observations -----------------------------
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05

        # Wire up key body names for AMP + critic.
        key_body_cfg = SceneEntityCfg(
            "robot", body_names=list(self.key_body_names), preserve_order=True
        )
        self.observations.amp.features.params["asset_cfg"] = key_body_cfg
        self.observations.critic.key_body_pos_b.params["asset_cfg"] = key_body_cfg

        # Match the gait_phase observation parameters to the gait reward
        # parameters below (cycle=0.5s, alternating L/R 180°, swing/stance 60/40).
        # Mismatches between obs phase and reward phase produce a confused policy.
        _gait_obs_params = {
            "gait_cycle": 0.5,
            "phase_offset_l": 0.0,
            "phase_offset_r": 0.5,
            "air_ratio_l": 0.6,
            "air_ratio_r": 0.6,
        }
        self.observations.policy.gait_phase.params = dict(_gait_obs_params)
        self.observations.critic.gait_phase.params = dict(_gait_obs_params)

        # ----------------------------- Actions -----------------------------
        self.actions.joint_pos.scale = 0.25

        # ----------------------------- Events -----------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.push_robot = None

        # Reference State Initialization (RSI).  Spawns the robot from a
        # near-stable frame of the LAFAN1 walks (filtered by base velocity
        # and root height) so the policy has to learn to *move while not
        # falling* instead of converging on the trivial "stand in default
        # pose" local optimum.  legged_lab/TienKung-Lab both rely on RSI to
        # break this local optimum.
        #
        # Critical: RSI must run AFTER ``reset_base`` and
        # ``reset_robot_joints`` so its absolute joint state and root pose
        # overwrite their uniform-noise initialization.  IsaacLab's
        # EventManager applies events in the order they appear on the cfg
        # class — assigning here (after super().__post_init__()) appends to
        # the end.  We also disable the uniform joint scaling so the RSI
        # joint pose isn't perturbed by 0.5–1.5× scaling.
        from isaaclab.managers import EventTermCfg as EventTerm
        self.events.reset_robot_joints = None  # RSI provides absolute joint state
        self.events.reset_from_ref = EventTerm(
            func=mdp.reset_from_reference_motion,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "height_offset": 0.05,
                # ~12% of LAFAN1 walk frames satisfy these — about 5500
                # near-stable frames, plenty for RSI sampling diversity
                # without spawning mid-stride with one foot airborne.
                "max_lin_vel_xy": 1.5,
                "max_ang_vel": 1.5,
                "min_root_height": 0.60,
            },
        )

        # ----------------------------- Rewards (legged_lab G1 style) -----------------------------
        # Disable old "is_alive" — replaced by negative termination_penalty
        # which is much stronger (legged_lab uses -50 for terminations).
        self.rewards.is_alive.weight = 0.0

        # Velocity-tracking rewards.  Dropped from 8.0/3.0 → 2.0/1.0 to match
        # TienKung-Lab.  Previously the policy could saturate this term just
        # by drifting forward without lifting feet (iter 10k: track=7.23/8.0,
        # feet_clearance=0.02/1.5 — clearance signal couldn't compete).
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Root penalties (legged_lab values)
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # legged_lab uses -1.0; we keep -1.0 too — combined with the strong
        # termination penalty, the policy learns to stay upright instead of
        # using flat_orientation as the only "don't fall" signal.
        self.rewards.flat_orientation_l2.weight = -2.0

        # Joint penalties (legged_lab values)
        self.rewards.joint_torques_l2.weight = -2.0e-6
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.joint_acc_l2.weight = -1.0e-7

        self.rewards.action_rate_l2.weight = -0.005

        # Contact rewards
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=self.foot_link_name
        )

        # ----------------------------- Gait periodic rewards -----------------------------
        # TienKung-Lab style: smooth phase clock (delta_t boundary blending) +
        # exp-shaped force/speed/clearance rewards.  Three signals together pin
        # the *cadence*, *contact*, and *swing-height* of the gait so the policy
        # can't just shuffle its feet along the floor.
        #
        # Phase layout (offsets 0.0 / 0.5 = 180° out of phase, single-stance gait):
        #   t in [0, air_ratio):  swing  → reward force≈0, speed-up, target z
        #   t in [air_ratio, 1):  stance → reward force>0, speed≈0
        # cycle 0.7s + air_ratio 0.4 → ~0.28s swing per foot (matches a normal walk).
        from isaaclab.managers import RewardTermCfg as RewTerm
        # TienKung-Lab gait parameters: 0.5s cycle, 60% swing.  Previously we
        # used 0.7s/40% which gave too few steps per second AND too short a
        # swing window — policy converged on shuffling because the foot
        # didn't have time to lift and the cadence was too slow to produce
        # the long alternating strides seen in the LAFAN1 motion clips.
        _gait_cycle = 0.5
        _gait_air_ratio = 0.6
        _offset_l = 0.0
        _offset_r = 0.5
        _foot_sensor = SceneEntityCfg("contact_forces", body_names=self.foot_link_name)
        _foot_asset = SceneEntityCfg("robot", body_names=self.foot_link_name)

        self.rewards.gait_frc_perio = RewTerm(
            func=mdp.gait_feet_frc_perio,
            weight=2.5,
            params={
                "sensor_cfg": _foot_sensor,
                "cycle": _gait_cycle,
                "offset_l": _offset_l,
                "offset_r": _offset_r,
                "air_ratio": _gait_air_ratio,
                "force_sigma": 200.0,
            },
        )
        self.rewards.gait_spd_perio = RewTerm(
            func=mdp.gait_feet_spd_perio,
            weight=0.3,
            params={
                "asset_cfg": _foot_asset,
                "cycle": _gait_cycle,
                "offset_l": _offset_l,
                "offset_r": _offset_r,
                "air_ratio": _gait_air_ratio,
                "speed_sigma": 0.5,
            },
        )
        self.rewards.gait_support_perio = RewTerm(
            func=mdp.gait_feet_frc_support_perio,
            weight=0.8,
            params={
                "sensor_cfg": _foot_sensor,
                "cycle": _gait_cycle,
                "offset_l": _offset_l,
                "offset_r": _offset_r,
                "air_ratio": _gait_air_ratio,
                "force_sigma": 150.0,
            },
        )

        # Foot clearance — bumped weight to 4.0 and target to 0.12m so the
        # antive linear-clip reward is in the same league as the velocity
        # tracker (8.0).  At iter 10k with weight=1.5 / target=0.10 the
        # reward was only 0.02, i.e. the foot was barely 1.6cm off the ground;
        # the policy had no incentive to lift higher because tracking already
        # paid 7.2 / 8.0.
        self.rewards.feet_clearance = RewTerm(
            func=mdp.feet_clearance,
            weight=4.0,
            params={
                "asset_cfg": _foot_asset,
                "cycle": _gait_cycle,
                "offset_l": _offset_l,
                "offset_r": _offset_r,
                "air_ratio": _gait_air_ratio,
                "target_height": 0.15,
                "height_sigma": 0.025,
            },
        )

        # Foot-spacing / slide constraints (TienKung-Lab style).
        self.rewards.feet_y_distance = RewTerm(
            func=mdp.feet_y_distance,
            weight=-2.0,
            params={
                "asset_cfg": _foot_asset,
                "target_distance": 0.20,
                "command_name": "base_velocity",
                "y_vel_threshold": 0.1,
            },
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.25,
            params={
                "sensor_cfg": _foot_sensor,
                "asset_cfg": _foot_asset,
                "contact_force_threshold": 1.0,
            },
        )

        # Termination penalty (legged_lab signature reward — single biggest
        # reason their policies don't fall over).  Heavy negative reward each
        # time the episode terminates from anything except time_out.
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated, weight=-200.0
        )

        # Joint deviation penalties (TienKung-style) — keeps arms in natural
        # position (prevents drooping) and hips/waist from flailing.
        self.rewards.joint_deviation_arms = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                        ".*_wrist_.*",
                    ],
                )
            },
        )
        self.rewards.joint_deviation_hip = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_yaw_joint",
                        ".*_hip_roll_joint",
                    ],
                )
            },
        )
        # Waist (yaw/roll/pitch) — strong penalty.  When this was bundled with
        # the hip term at weight -0.1, the policy let the waist pitch drift
        # backwards into a "lean-back" posture.  Pull it out and weight it
        # ~5× heavier so the torso stays upright.
        self.rewards.joint_deviation_waist = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "waist_yaw_joint",
                        "waist_roll_joint",
                        "waist_pitch_joint",
                    ],
                )
            },
        )

        # ----------------------------- Terminations (legged_lab G1 style) -----------------------------
        # legged_lab disables base contact for G1 — humanoids fall in many ways
        # not always involving torso contact.  Replace with height + orientation
        # checks which are more reliable for early termination.
        self.terminations.illegal_contact = None

        from isaaclab.managers import TerminationTermCfg as DoneTerm
        # Drop reference_deviation — legged_lab doesn't use it; the strong
        # termination_penalty + height/orientation checks already do this job.
        self.terminations.reference_deviation = None

        # Height-based termination: episode dies if base falls below 0.2m
        # (matches legged_lab default).  Note this is the **pelvis** height
        # for G1, which sits much lower than the head; standing height ≈ 0.8m,
        # so 0.2m means clearly fallen.
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.2},
        )
        # Orientation termination: episode dies if base tilts > 75° from
        # upright.  legged_lab uses 60° but their RSI + height_offset combo
        # gets the robot into much better starting poses than ours; until we
        # match that we keep this looser.
        import math as _math
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": _math.radians(75.0)},
        )

        # ----------------------------- Commands -----------------------------
        # walk1_subject1.npz contains only forward / turning gait — there is
        # no backward-walking demonstration.  Sampling negative lin_vel_x
        # forces the policy to invent a motion the discriminator has never
        # seen, which collapses to "shuffle backwards in place".  Restrict to
        # the directions the dataset actually covers.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ----------------------------- AMP Motion Data -----------------------------
        # Default to one validated walk clip.  A directory path will load every
        # supported motion file under it recursively, so we keep the built-in
        # default narrow and predictable here.  For broader training, override
        # this from the CLI with ``--motion_file <walk_only_dir_or_npz>``.
        self.robot_type = "g1"
        self.amp_motion_files = os.path.join(
            LEGGED_RL_LAB_ROOT_DIR,
            "data", "motion", "LAFAN1_Retargeting_Dataset", "g1", "walk1_subject1.npz",
        )

@configclass
class UnitreeG1AMPFlatEnvCfg_PLAY(UnitreeG1AMPFlatEnvCfg):
    """Unitree G1 AMP environment for visualisation / play."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.lin_vel_cmd_levels = None
