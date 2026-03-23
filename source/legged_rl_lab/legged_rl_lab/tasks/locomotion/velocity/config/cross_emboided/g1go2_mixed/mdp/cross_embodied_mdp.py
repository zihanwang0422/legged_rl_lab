# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Custom MDP terms for the cross-embodied G1 + Go2 mixed scene.

Every function takes the ``env: ManagerBasedRLEnv`` as first argument so it
can be used directly with ``ObsTerm``, ``RewTerm``, ``DoneTerm`` and the
custom ``CrossEmbodiedJointPosAction``.

Observation helpers
-------------------
These functions select data from the **active** robot per environment:
- Environments [0 .. n_g1-1]    → ``robot_g1`` is active.
- Environments [n_g1 .. n-1]    → ``robot_go2`` is active.

The helper ``env.is_g1_env`` (set by ``CrossEmbodiedG1Go2Env.__init__``) is
a boolean tensor of shape ``(num_envs,)`` that encodes which robot is active.
"""

from __future__ import annotations

import math
import re as _re
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------


def robot_type_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """2-dim one-hot robot type: ``[1, 0]`` = G1, ``[0, 1]`` = Go2.

    Shape: ``(num_envs, 2)``.
    """
    return env.robot_type_onehot


def base_ang_vel_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Base angular velocity in body frame from the active robot.

    Shape: ``(num_envs, 3)``.
    """
    g1_vel = env.scene["robot_g1"].data.root_ang_vel_b   # (N, 3)
    go2_vel = env.scene["robot_go2"].data.root_ang_vel_b  # (N, 3)
    mask = env.is_g1_env.unsqueeze(-1)                    # (N, 1)
    return torch.where(mask, g1_vel, go2_vel)


def projected_gravity_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gravity vector projected onto the active robot's body frame.

    Shape: ``(num_envs, 3)``.
    """
    g1_grav = env.scene["robot_g1"].data.projected_gravity_b
    go2_grav = env.scene["robot_go2"].data.projected_gravity_b
    mask = env.is_g1_env.unsqueeze(-1)
    return torch.where(mask, g1_grav, go2_grav)


def base_lin_vel_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Base linear velocity in body frame from the active robot (for critic).

    Shape: ``(num_envs, 3)``.
    """
    g1_vel = env.scene["robot_g1"].data.root_lin_vel_b
    go2_vel = env.scene["robot_go2"].data.root_lin_vel_b
    mask = env.is_g1_env.unsqueeze(-1)
    return torch.where(mask, g1_vel, go2_vel)


def joint_pos_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Relative joint positions, zero-padded to G1's DOF count (29).

    - G1 envs: all 29 joint offsets.
    - Go2 envs: 12 joint offsets followed by 17 zeros.

    Shape: ``(num_envs, n_g1_joints)``.
    """
    g1_pos = (
        env.scene["robot_g1"].data.joint_pos
        - env.scene["robot_g1"].data.default_joint_pos
    )  # (N, 29)
    go2_pos = (
        env.scene["robot_go2"].data.joint_pos
        - env.scene["robot_go2"].data.default_joint_pos
    )  # (N, 12)
    pad_len = g1_pos.shape[-1] - go2_pos.shape[-1]
    go2_padded = F.pad(go2_pos, (0, pad_len))  # (N, 29)
    mask = env.is_g1_env.unsqueeze(-1)
    return torch.where(mask, g1_pos, go2_padded)


def joint_vel_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Joint velocities, zero-padded to G1's DOF count (29).

    Shape: ``(num_envs, n_g1_joints)``.
    """
    g1_vel = env.scene["robot_g1"].data.joint_vel   # (N, 29)
    go2_vel = env.scene["robot_go2"].data.joint_vel  # (N, 12)
    pad_len = g1_vel.shape[-1] - go2_vel.shape[-1]
    go2_padded = F.pad(go2_vel, (0, pad_len))
    mask = env.is_g1_env.unsqueeze(-1)
    return torch.where(mask, g1_vel, go2_padded)


def last_action_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last applied action; dims beyond Go2's DOF count are zeroed for Go2 envs.

    Shape: ``(num_envs, n_g1_joints)``.
    """
    action = env.action_manager.action.clone()  # (N, 29)
    n_go2 = env.scene["robot_go2"].data.joint_pos.shape[-1]  # 12
    # Zero out unused dims for Go2 envs so the policy sees a clean signal.
    action[~env.is_g1_env, n_go2:] = 0.0
    return action


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def track_lin_vel_xy_cross(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Gaussian velocity-tracking reward for the active robot (XY plane).

    Shape: ``(num_envs,)``.
    """
    g1_vel = env.scene["robot_g1"].data.root_lin_vel_b[:, :2]   # (N, 2)
    go2_vel = env.scene["robot_go2"].data.root_lin_vel_b[:, :2]  # (N, 2)
    cmd = env.command_manager.get_command(command_name)[:, :2]   # (N, 2)

    mask = env.is_g1_env.unsqueeze(-1)
    vel = torch.where(mask, g1_vel, go2_vel)

    error = torch.sum(torch.square(cmd - vel), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_cross(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Gaussian yaw-rate tracking reward for the active robot.

    Shape: ``(num_envs,)``.
    """
    g1_yaw = env.scene["robot_g1"].data.root_ang_vel_b[:, 2]
    go2_yaw = env.scene["robot_go2"].data.root_ang_vel_b[:, 2]
    cmd_yaw = env.command_manager.get_command(command_name)[:, 2]  # (N,)

    yaw = torch.where(env.is_g1_env, g1_yaw, go2_yaw)
    error = torch.square(cmd_yaw - yaw)
    return torch.exp(-error / std**2)


def is_alive_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns 1 for envs whose active robot has not terminated yet.

    Shape: ``(num_envs,)``.
    """
    # env.termination_manager.terminated is (N,) bool
    return (~env.termination_manager.terminated).float()


def lin_vel_z_l2_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared Z-velocity penalty for the active robot.

    Shape: ``(num_envs,)``.
    """
    g1_vz = env.scene["robot_g1"].data.root_lin_vel_b[:, 2]
    go2_vz = env.scene["robot_go2"].data.root_lin_vel_b[:, 2]
    vz = torch.where(env.is_g1_env, g1_vz, go2_vz)
    return torch.square(vz)


def ang_vel_xy_l2_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared roll/pitch rate penalty for the active robot.

    Shape: ``(num_envs,)``.
    """
    g1_w = env.scene["robot_g1"].data.root_ang_vel_b[:, :2]
    go2_w = env.scene["robot_go2"].data.root_ang_vel_b[:, :2]
    mask = env.is_g1_env.unsqueeze(-1)
    w = torch.where(mask, g1_w, go2_w)
    return torch.sum(torch.square(w), dim=1)


def flat_orientation_l2_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared projected-gravity XY penalty (penalises tilting).

    Shape: ``(num_envs,)``.
    """
    g1_grav = env.scene["robot_g1"].data.projected_gravity_b[:, :2]
    go2_grav = env.scene["robot_go2"].data.projected_gravity_b[:, :2]
    mask = env.is_g1_env.unsqueeze(-1)
    grav = torch.where(mask, g1_grav, go2_grav)
    return torch.sum(torch.square(grav), dim=1)


def joint_vel_l2_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sum of squared joint velocities for the active robot.

    Shape: ``(num_envs,)``.
    """
    g1_jv = torch.sum(torch.square(env.scene["robot_g1"].data.joint_vel), dim=1)
    go2_jv = torch.sum(torch.square(env.scene["robot_go2"].data.joint_vel), dim=1)
    return torch.where(env.is_g1_env, g1_jv, go2_jv)


def joint_acc_l2_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sum of squared joint accelerations for the active robot.

    Shape: ``(num_envs,)``.
    """
    g1_ja = torch.sum(torch.square(env.scene["robot_g1"].data.joint_acc), dim=1)
    go2_ja = torch.sum(torch.square(env.scene["robot_go2"].data.joint_acc), dim=1)
    return torch.where(env.is_g1_env, g1_ja, go2_ja)


def action_rate_l2_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sum of squared action differences between consecutive steps.

    Shape: ``(num_envs,)``.

    Uses ``env.action_manager.action`` and the previous step cached in
    ``env._cross_prev_action`` (initialised lazily on first call).
    """
    cur = env.action_manager.action   # (N, 29)
    if not hasattr(env, "_cross_prev_action"):
        env._cross_prev_action = cur.clone()
    diff = cur - env._cross_prev_action
    env._cross_prev_action = cur.clone()

    # Zero out inactive dims for Go2 envs so penalty is fair.
    n_go2 = env.scene["robot_go2"].data.joint_pos.shape[-1]
    diff_masked = diff.clone()
    diff_masked[~env.is_g1_env, n_go2:] = 0.0
    return torch.sum(torch.square(diff_masked), dim=1)


def joint_pos_limits_cross(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Soft joint-limit penalty for the active robot.

    Returns the sum of soft-limit violations (>0 means outside limits).
    Shape: ``(num_envs,)``.
    """

    def _limits(robot_name: str) -> torch.Tensor:
        robot = env.scene[robot_name]
        pos = robot.data.joint_pos
        lo, hi = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
        return torch.sum(
            (torch.clamp(pos - hi, min=0.0) + torch.clamp(lo - pos, min=0.0)),
            dim=1,
        )

    g1_pen = _limits("robot_g1")
    go2_pen = _limits("robot_go2")
    return torch.where(env.is_g1_env, g1_pen, go2_pen)


# ---------------------------------------------------------------------------
# Termination terms
# ---------------------------------------------------------------------------


def base_below_threshold_cross(
    env: ManagerBasedRLEnv,
    min_height: float = 0.25,
) -> torch.Tensor:
    """Terminate when the active robot's base falls below *min_height* above ground.

    Shape: ``(num_envs,)`` bool.
    """
    env_z = env.scene.env_origins[:, 2]  # (N,)
    g1_h = env.scene["robot_g1"].data.root_pos_w[:, 2] - env_z
    go2_h = env.scene["robot_go2"].data.root_pos_w[:, 2] - env_z
    h = torch.where(env.is_g1_env, g1_h, go2_h)
    return h < min_height


def bad_orientation_cross(
    env: ManagerBasedRLEnv,
    limit_angle: float = 0.8,
) -> torch.Tensor:
    """Terminate when gravity projection angle exceeds *limit_angle* rad.

    A large gravity XY projection means the robot is tilting severely.
    Shape: ``(num_envs,)`` bool.
    """
    g1_grav = env.scene["robot_g1"].data.projected_gravity_b
    go2_grav = env.scene["robot_go2"].data.projected_gravity_b
    mask = env.is_g1_env.unsqueeze(-1)
    grav = torch.where(mask, g1_grav, go2_grav)
    return torch.acos(-grav[:, 2].clamp(-1.0, 1.0)) > limit_angle


def illegal_contact_cross(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate on illegal contact for the active robot.

    The contact sensors must be named ``contact_forces_g1`` and
    ``contact_forces_go2`` in the scene.  Only non-foot bodies are
    penalised (the sensor prim paths already specify this via regex).

    Shape: ``(num_envs,)`` bool.
    """
    g1_force = torch.max(
        torch.norm(env.scene["contact_forces_g1"].data.net_forces_w, dim=-1), dim=-1
    ).values  # (N,)
    go2_force = torch.max(
        torch.norm(env.scene["contact_forces_go2"].data.net_forces_w, dim=-1), dim=-1
    ).values  # (N,)
    force = torch.where(env.is_g1_env, g1_force, go2_force)
    return force > threshold


# ---------------------------------------------------------------------------
# Custom action term
# ---------------------------------------------------------------------------


class CrossEmbodiedJointPosAction(ActionTerm):
    """Routes padded joint-position targets to the active robot per environment.

    The policy outputs a unified action tensor of shape
    ``(num_envs, n_g1_joints)`` (i.e. 29 dims for G1 + Go2).
    - G1 envs: all 29 dims are applied to ``robot_g1``.
    - Go2 envs: first 12 dims are applied to ``robot_go2``; dims 12-28 are
      discarded (the policy learns to zero them via the ``robot_id`` obs).

    The inactive robot (parked at altitude) receives its default joint targets
    so it remains still.
    """

    cfg: "CrossEmbodiedJointPosActionCfg"

    def __init__(self, cfg: "CrossEmbodiedJointPosActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._robot_g1 = self._env.scene["robot_g1"]
        self._robot_go2 = self._env.scene["robot_go2"]

        # Resolve joint IDs (all joints for each robot)
        self._joint_ids_g1, _ = self._robot_g1.find_joints(".*")
        self._joint_ids_go2, _ = self._robot_go2.find_joints(".*")
        self._n_g1 = len(self._joint_ids_g1)  # 29
        self._n_go2 = len(self._joint_ids_go2)  # 12

        # Storage for processed targets
        num_envs = env.num_envs
        self._g1_targets = torch.zeros(num_envs, self._n_g1, device=env.device)
        self._go2_targets = torch.zeros(num_envs, self._n_go2, device=env.device)
        # Last scaled actions for last_action obs (inactive dims zeroed)
        self._last_scaled = torch.zeros(num_envs, self._n_g1, device=env.device)

    # ------------- ActionTerm abstract interface ---------------------------

    @property
    def action_dim(self) -> int:
        return self._n_g1  # 29 – the maximum (G1) DOF count

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._last_scaled

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._last_scaled

    def process_actions(self, actions: torch.Tensor) -> None:
        """Scale raw policy actions and compute per-robot joint targets."""
        is_g1 = self._env.is_g1_env  # (N,) bool

        # Per-env scales
        scale = torch.where(
            is_g1,
            torch.full((actions.shape[0],), self.cfg.scale_g1, device=actions.device),
            torch.full((actions.shape[0],), self.cfg.scale_go2, device=actions.device),
        )
        scaled = actions * scale.unsqueeze(-1)  # (N, 29)

        # Targets = default_pos + scaled_action
        self._g1_targets = self._robot_g1.data.default_joint_pos + scaled[:, : self._n_g1]
        self._go2_targets = self._robot_go2.data.default_joint_pos + scaled[:, : self._n_go2]

        # Store masked version for last_action obs
        self._last_scaled = scaled.clone()
        self._last_scaled[~is_g1, self._n_go2 :] = 0.0

    def apply_actions(self) -> None:
        """Apply targets to the active robot; hold inactive robot at defaults."""
        is_g1 = self._env.is_g1_env
        g1_ids = is_g1.nonzero(as_tuple=False).view(-1)
        go2_ids = (~is_g1).nonzero(as_tuple=False).view(-1)

        # Active robot targets
        if len(g1_ids) > 0:
            self._robot_g1.set_joint_position_target(
                self._g1_targets[g1_ids],
                joint_ids=self._joint_ids_g1,
                env_ids=g1_ids,
            )
        if len(go2_ids) > 0:
            self._robot_go2.set_joint_position_target(
                self._go2_targets[go2_ids],
                joint_ids=self._joint_ids_go2,
                env_ids=go2_ids,
            )

        # Inactive robot: hold at default (parked robots should not drift)
        if len(go2_ids) > 0:
            self._robot_g1.set_joint_position_target(
                self._robot_g1.data.default_joint_pos[go2_ids],
                joint_ids=self._joint_ids_g1,
                env_ids=go2_ids,
            )
        if len(g1_ids) > 0:
            self._robot_go2.set_joint_position_target(
                self._robot_go2.data.default_joint_pos[g1_ids],
                joint_ids=self._joint_ids_go2,
                env_ids=g1_ids,
            )


@configclass
class CrossEmbodiedJointPosActionCfg(ActionTermCfg):
    """Configuration for :class:`CrossEmbodiedJointPosAction`."""

    class_type: type = CrossEmbodiedJointPosAction

    # asset_name is required by ActionTermCfg but unused here (robots are
    # accessed directly via env.scene["robot_g1"] / env.scene["robot_go2"]).
    asset_name: str = "robot_g1"

    scale_g1: float = 0.5
    """Action scale applied to G1 envs (scales raw policy output before adding
    to default joint positions)."""

    scale_go2: float = 0.25
    """Action scale applied to Go2 envs."""


# ---------------------------------------------------------------------------
# G1-only joint deviation reward
# ---------------------------------------------------------------------------


def joint_deviation_g1_l1_cross(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
) -> torch.Tensor:
    """Joint deviation L1 for specified G1 joints only; Go2 envs return 0.

    Applies L1 penalty on the deviation of G1's specified joints from their
    default positions.  Go2 envs receive zero reward for this term, so the
    penalty is effectively G1-specific without needing separate reward configs.

    Args:
        joint_names: List of joint name regex patterns (fullmatch) for robot_g1.
                     Example: ``["waist.*", ".*_hip_roll_joint"]``.
    """
    # Lazy-cache resolved joint index tensor per unique joint_names tuple.
    cache_key = f"_g1_dev_ids_{hash(tuple(joint_names))}"
    if not hasattr(env, cache_key):
        g1 = env.scene["robot_g1"]
        all_names = g1.data.joint_names
        ids = [
            i for i, name in enumerate(all_names)
            if any(_re.fullmatch(pat, name) for pat in joint_names)
        ]
        setattr(env, cache_key, torch.tensor(ids, device=env.device, dtype=torch.long))

    joint_ids = getattr(env, cache_key)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    g1 = env.scene["robot_g1"]
    pos = g1.data.joint_pos[:, joint_ids]
    default = g1.data.default_joint_pos[:, joint_ids]
    deviation = torch.sum(torch.abs(pos - default), dim=-1)  # (N,)
    # Zero out for Go2 envs so the penalty only affects G1 training.
    return deviation * env.is_g1_env.float()
