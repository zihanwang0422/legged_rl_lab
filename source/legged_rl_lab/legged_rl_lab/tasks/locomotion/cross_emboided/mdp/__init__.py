# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for cross-embodied locomotion tasks.

Re-exports isaaclab base mdp functions and the project-level custom locomotion
functions so that cross-embodied configs can import exclusively from this
module without depending on any velocity-task module.
"""

# ── isaaclab base MDP (observations, rewards, events, terminations, etc.) ──
from isaaclab.envs.mdp import *  # noqa: F401, F403

# ── isaaclab_tasks locomotion velocity MDP (standard loco rewards/events) ──
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# ── project-level custom locomotion MDP functions ──
from legged_rl_lab.tasks.locomotion.velocity.mdp.commands import *  # noqa: F401, F403
from legged_rl_lab.tasks.locomotion.velocity.mdp.curriculums import *  # noqa: F401, F403
from legged_rl_lab.tasks.locomotion.velocity.mdp.events import *  # noqa: F401, F403
from legged_rl_lab.tasks.locomotion.velocity.mdp.observations import *  # noqa: F401, F403
from legged_rl_lab.tasks.locomotion.velocity.mdp.rewards import *  # noqa: F401, F403
from legged_rl_lab.tasks.locomotion.velocity.mdp.utils import *  # noqa: F401, F403

# ── cross-embodied specific MDP terms ──────────────────────────────────────
from .procedural_obs import (  # noqa: F401
    phase,
    morphology_params,
    modify_procedural_articulations,
    setup_morphology_params,
    setup_cross_embodied_morphology_params,
    ProceduralRobotEnv,
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
