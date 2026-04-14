# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for cross-embodied locomotion tasks.

Self-contained: does NOT import from legged_rl_lab.tasks.locomotion.velocity.
All custom functions are maintained locally under this package.
"""

# ── isaaclab base MDP (observations, rewards, events, terminations, etc.) ──
from isaaclab.envs.mdp import *  # noqa: F401, F403

# ── isaaclab_tasks locomotion velocity MDP (standard loco rewards/events) ──
# (external library — not the project velocity task)
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# ── cross-embodied local MDP functions (independent copies) ──────────────
from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403

# ── cross-embodied specific MDP terms ──────────────────────────────────────
from .procedural_obs import (  # noqa: F401
    phase,
    morphology_params,
    modify_procedural_articulations,
    setup_morphology_params,
    setup_humanoid_morphology_params,
    setup_quadruped_morphology_params,
    setup_cross_embodied_morphology_params,
    ProceduralRobotEnv,
    ProceduralHumanoidRobotEnv,
    ProceduralQuadrupedRobotEnv,
)

from .cross_procedural_mdp import (  # noqa: F401
    ObsLayout,
    CrossEmbodiedEncoderCfg,
    CrossProceduralEnv,
    MaskEncoder,
    TransformerObsEncoder,
    GCNObsEncoder,
    build_obs_encoder,
    ActorCriticWithEncoder,
    register_in_rsl_rl,
    # ── Mixed biped+quad dispatch ────────────────────────────────────────
    ProceduralMixedRobotEnv,
    ProceduralMixedJointPosAction,
    ProceduralMixedJointPosActionCfg,
    setup_mixed_morphology_params,
    setup_mixed_two_entity_morphology_params,
    modify_mixed_procedural_articulations,
    base_ang_vel_mixed,
    projected_gravity_mixed,
    base_lin_vel_mixed,
    joint_pos_mixed,
    joint_vel_mixed,
    last_action_mixed,
    track_lin_vel_xy_exp_mixed,
    track_ang_vel_z_exp_mixed,
    ang_vel_xy_l2_mixed,
    flat_orientation_l2_mixed,
    body_roll_l2_mixed,
    joint_torques_l2_mixed,
    joint_vel_l2_mixed,
    joint_acc_l2_mixed,
    joint_pos_limits_mixed,
    action_rate_l2_mixed,
    stand_still_mixed,
    undesired_contacts_mixed,
    feet_air_time_mixed,
    feet_slide_mixed,
    illegal_contact_base_mixed,
)

from .cross_embodied_mdp import (  # noqa: F401
    # ---- observations ----
    robot_type_id,
    base_ang_vel_cross,
    projected_gravity_cross,
    base_lin_vel_cross,
    joint_pos_cross,
    joint_vel_cross,
    last_action_cross,
    # ---- rewards ----
    track_lin_vel_xy_cross,
    track_ang_vel_z_cross,
    is_alive_cross,
    lin_vel_z_l2_cross,
    ang_vel_xy_l2_cross,
    flat_orientation_l2_cross,
    joint_vel_l2_cross,
    joint_acc_l2_cross,
    action_rate_l2_cross,
    joint_pos_limits_cross,
    joint_deviation_g1_l1_cross,
    # ---- additional rewards ----
    base_height_g1_cross,
    gait_g1_cross,
    feet_air_time_go2_cross,
    stand_still_cross,
    # ---- G1-specific feet ----
    feet_slide_g1_cross,
    feet_clearance_g1_cross,
    feet_air_time_g1_cross,
    undesired_contacts_g1_cross,
    # ---- Go2-specific feet / contact ----
    feet_slide_go2_cross,
    undesired_contacts_go2_cross,
    joint_torques_go2_cross,
    # ---- terminations ----
    base_below_threshold_cross,
    bad_orientation_cross,
    illegal_contact_cross,
    # ---- action term ----
    CrossEmbodiedJointPosActionCfg,
    CrossEmbodiedJointPosAction,
)
