# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for the cross-embodied G1 + Go2 mixed scene."""

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
    # ---- terminations ----
    base_below_threshold_cross,
    bad_orientation_cross,
    illegal_contact_cross,
    # ---- action term ----
    CrossEmbodiedJointPosActionCfg,
    CrossEmbodiedJointPosAction,
)
