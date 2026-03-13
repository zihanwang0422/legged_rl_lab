# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Motion data loader for AMP reference motions.

Supports multiple source formats:

1. **Pre-processed AMP features** (.npy / .pt / .pth)
   - numpy array or torch tensor of shape ``(N, obs_dim)``
   - dict with key ``"amp_obs"`` containing such an array

2. **AMASS-retargeted NPZ** (.npz) — e.g. AMASS_Retargeted_for_G1
   Keys: ``fps``, ``dof_names``, ``dof_positions``, ``dof_velocities``,
   ``body_positions``, ``body_rotations``, ``body_linear_velocities``,
   ``body_angular_velocities``.
   The loader extracts AMP features:
     joint_pos_rel | joint_vel | base_lin_vel_b | base_ang_vel_b | foot_pos_b

3. **LAFAN1 Retargeting Dataset CSV** (.csv) — 30 FPS, per-robot column layout
   Each row: ``[x, y, z, qx, qy, qz, qw,  joint_0 … joint_N-1]``
   Velocities are computed via finite difference.
   Foot positions are unavailable (set to zero) without running FK.

4. **Directory** — loads all supported files recursively and concatenates.

AMP observation layout (matches ``AMPCfg`` in amp_env_cfg.py):
  joint_pos_rel (num_dof) | joint_vel (num_dof) |
  base_lin_vel (3) | base_ang_vel (3) | foot_positions_base (num_foot_links * 3)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known robot profiles
# ---------------------------------------------------------------------------
# foot_body_indices: indices into body_names array (from .npz)
# default_joint_pos: per-joint default positions matching dof_names order
# csv_quat_format: "xyzw" (LAFAN1) or "wxyz"
# joint_reorder_map: MuJoCo/AMASS order → IsaacLab BFS order (for .npz files)
_ROBOT_PROFILES: dict[str, dict] = {
    "g1": {
        "num_dof": 29,
        "foot_body_indices": [6, 12],  # left_ankle_roll_link, right_ankle_roll_link
        # G1 29-DOF default standing posture (matches UNITREE_G1_29DOF_CFG BFS order)
        "default_joint_pos": np.array([
            -0.1,  # [0] left_hip_pitch_joint
            -0.1,  # [1] right_hip_pitch_joint
             0.0,  # [2] waist_yaw_joint
             0.0,  # [3] left_hip_roll_joint
             0.0,  # [4] right_hip_roll_joint
             0.0,  # [5] waist_roll_joint
             0.0,  # [6] left_hip_yaw_joint
             0.0,  # [7] right_hip_yaw_joint
             0.0,  # [8] waist_pitch_joint
             0.3,  # [9] left_knee_joint
             0.3,  # [10] right_knee_joint
             0.3,  # [11] left_shoulder_pitch_joint
             0.3,  # [12] right_shoulder_pitch_joint
            -0.2,  # [13] left_ankle_pitch_joint
            -0.2,  # [14] right_ankle_pitch_joint
             0.25, # [15] left_shoulder_roll_joint
            -0.25, # [16] right_shoulder_roll_joint
             0.0,  # [17] left_ankle_roll_joint
             0.0,  # [18] right_ankle_roll_joint
             0.0,  # [19] left_shoulder_yaw_joint
             0.0,  # [20] right_shoulder_yaw_joint
             0.97, # [21] left_elbow_joint
             0.97, # [22] right_elbow_joint
             0.15, # [23] left_wrist_roll_joint
            -0.15, # [24] right_wrist_roll_joint
             0.0,  # [25] left_wrist_pitch_joint
             0.0,  # [26] right_wrist_pitch_joint
             0.0,  # [27] left_wrist_yaw_joint
             0.0,  # [28] right_wrist_yaw_joint
        ], dtype=np.float32),
        "csv_quat_format": "xyzw",
        # MuJoCo/AMASS joint order → IsaacLab BFS order
        # AMASS order: left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, ...
        # IsaacLab BFS: left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, ...
        "joint_reorder_map": np.array([
            0, 3, 6, 9, 13, 17,  # left leg: hip_p/r/y, knee, ankle_p/r
            1, 4, 7, 10, 14, 18,  # right leg
            2, 5, 8,              # waist: yaw, roll, pitch
            11, 15, 19, 21, 23, 25, 27,  # left arm
            12, 16, 20, 22, 24, 26, 28,  # right arm
        ], dtype=np.int32),
    },
    "go2": {
        "num_dof": 12,
        "foot_body_indices": [3, 6, 9, 12],  # FL, FR, RL, RR foot
        "default_joint_pos": np.zeros(12, dtype=np.float32),
        "csv_quat_format": "wxyz",
        "joint_reorder_map": None,  # Go2 uses same order (or verify if needed)
    },
}


# ---------------------------------------------------------------------------
# Quaternion math (numpy, no IsaacLab dependency)
# Convention: [w, x, y, z]
# ---------------------------------------------------------------------------

def _quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by the *inverse* of quaternion ``q``.

    Args:
        q: ``(N, 4)`` quaternions in ``[w, x, y, z]`` order.
        v: ``(N, 3)`` vectors.

    Returns:
        ``(N, 3)`` rotated vectors.
    """
    q_w = q[:, 0:1]
    q_xyz = -q[:, 1:4]          # conjugate = inverse for unit quat
    uv = np.cross(q_xyz, v)     # (N, 3)
    uuv = np.cross(q_xyz, uv)   # (N, 3)
    return v + 2.0 * (q_w * uv + uuv)


def _finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    """Central/forward finite difference along axis 0."""
    vel = np.empty_like(x)
    vel[:-1] = (x[1:] - x[:-1]) * fps
    vel[-1] = vel[-2]           # repeat last frame
    return vel


def _ang_vel_from_quats(q: np.ndarray, fps: float) -> np.ndarray:
    """Estimate angular velocity (world frame) from consecutive quaternions.

    Uses: ω ≈ 2 * q_conj ⊗ dq/dt  (imaginary part only)
    Args:
        q: ``(N, 4)`` quaternions ``[w, x, y, z]``.
    Returns:
        ``(N, 3)`` angular velocity in *local* frame.
    """
    dq = np.empty_like(q)
    dq[:-1] = (q[1:] - q[:-1]) * fps
    dq[-1] = dq[-2]
    # q_conj ⊗ dq: imaginary part → angular velocity in body frame
    q_w = q[:, 0:1]
    q_xyz = q[:, 1:4]
    dq_w = dq[:, 0:1]
    dq_xyz = dq[:, 1:4]
    # imaginary part of (q_conj ⊗ dq): 2*(q_w*dq_xyz - dq_w*q_xyz - q_xyz×dq_xyz)
    ang_vel = 2.0 * (q_w * dq_xyz - dq_w * q_xyz + np.cross(-q_xyz, dq_xyz))
    return ang_vel.astype(np.float32)


# ---------------------------------------------------------------------------
# MotionLoader
# ---------------------------------------------------------------------------

class MotionLoader:
    """Loads and stores reference motion data for AMP training.

    Parameters
    ----------
    device:
        Torch device for the output tensor.
    robot:
        Robot identifier for built-in profiles (``"g1"``, ``"go2"``).
        Used when loading raw motion files (.npz / .csv) that need conversion
        to AMP feature vectors.  Ignored for pre-processed ``.npy`` / ``.pt``
        files that already contain an ``"amp_obs"`` array.
    default_joint_pos:
        Override the default joint positions used for ``joint_pos_rel``
        computation.  Shape: ``(num_dof,)``.
    """

    def __init__(
        self,
        device: str = "cpu",
        robot: str = "g1",
        default_joint_pos: Optional[np.ndarray] = None,
    ) -> None:
        self.device = device
        self.robot = robot.lower()
        profile = _ROBOT_PROFILES.get(self.robot, {})
        self._profile = profile
        if default_joint_pos is not None:
            self._default_jpos = np.asarray(default_joint_pos, dtype=np.float32)
        else:
            self._default_jpos = profile.get("default_joint_pos", None)
        self.data: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> torch.Tensor:
        """Load motion data from a file or directory.

        Dispatches to the appropriate sub-loader based on file extension.

        Args:
            path: Path to a motion file or directory.

        Returns:
            AMP observation tensor. Shape: ``(total_frames, obs_dim)``
        """
        if os.path.isdir(path):
            return self._load_directory(path)
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".npy":
            return self._load_npy(path)
        elif ext in (".pt", ".pth"):
            return self._load_pt(path)
        elif ext == ".npz":
            return self._load_npz(path)
        elif ext == ".csv":
            return self._load_csv(path)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}': {path}. "
                "Supported: .npy, .pt, .pth, .npz, .csv, or a directory."
            )

    # ------------------------------------------------------------------
    # Per-format loaders
    # ------------------------------------------------------------------

    def _load_npy(self, path: str) -> torch.Tensor:
        """Load pre-processed AMP features from a .npy file."""
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        if isinstance(data, dict):
            if "amp_obs" in data:
                data = data["amp_obs"]
            else:
                raise KeyError(f"Expected key 'amp_obs' in dict, got: {list(data.keys())}")
        tensor = torch.from_numpy(np.array(data, dtype=np.float32)).to(self.device)
        self.data = tensor
        return tensor

    def _load_pt(self, path: str) -> torch.Tensor:
        """Load pre-processed AMP features from a .pt/.pth file."""
        data = torch.load(path, weights_only=True, map_location=self.device)
        if isinstance(data, dict):
            if "amp_obs" in data:
                data = data["amp_obs"]
            else:
                raise KeyError(f"Expected key 'amp_obs' in dict, got: {list(data.keys())}")
        tensor = data.float().to(self.device)
        self.data = tensor
        return tensor

    def _load_npz(self, path: str) -> torch.Tensor:
        """Load AMASS-retargeted .npz and convert to flat AMP feature vectors.

        Expected keys: ``dof_positions``, ``dof_velocities``,
        ``body_positions``, ``body_rotations``,
        ``body_linear_velocities``, ``body_angular_velocities``.

        Quaternion convention in the file: ``[w, x, y, z]``.

        **Joint reordering**: AMASS data uses MuJoCo joint order, but IsaacLab
        uses BFS traversal order. This method automatically reorders joints
        if a ``joint_reorder_map`` is defined in the robot profile.

        AMP feature layout:
          joint_pos_rel(num_dof) | joint_vel(num_dof) |
          base_lin_vel_b(3) | base_ang_vel_b(3) | foot_pos_b(num_feet*3)
        """
        d = np.load(path, allow_pickle=True)
        dof_pos = d["dof_positions"].astype(np.float32)    # (N, J) [MuJoCo order]
        dof_vel = d["dof_velocities"].astype(np.float32)   # (N, J) [MuJoCo order]
        body_pos = d["body_positions"].astype(np.float32)  # (N, B, 3)
        body_rot = d["body_rotations"].astype(np.float32)  # (N, B, 4) [w,x,y,z]
        body_lin = d["body_linear_velocities"].astype(np.float32)  # (N, B, 3)
        body_ang = d["body_angular_velocities"].astype(np.float32)  # (N, B, 3)

        N = dof_pos.shape[0]

        # Reorder joints from MuJoCo/AMASS order to IsaacLab BFS order
        reorder_map = self._profile.get("joint_reorder_map", None)
        if reorder_map is not None:
            dof_pos = dof_pos[:, reorder_map]  # (N, J) reordered
            dof_vel = dof_vel[:, reorder_map]  # (N, J) reordered
            logger.debug(f"Reordered joints from MuJoCo to IsaacLab BFS order (map: {reorder_map.tolist()})")

        # 1. joint_pos_rel
        if self._default_jpos is not None:
            jpos_rel = dof_pos - self._default_jpos[None, :]
        else:
            jpos_rel = dof_pos

        # 2. joint_vel
        jvel = dof_vel

        # 3. base linear / angular velocity in base frame
        root_quat = body_rot[:, 0, :]     # (N, 4) [w,x,y,z]
        base_lin_vel = _quat_rotate_inverse_np(root_quat, body_lin[:, 0, :])
        base_ang_vel = _quat_rotate_inverse_np(root_quat, body_ang[:, 0, :])

        # 4. foot positions in base frame
        foot_indices = self._profile.get("foot_body_indices", [])
        root_pos = body_pos[:, 0, :]      # (N, 3)
        foot_parts = []
        for idx in foot_indices:
            rel_w = body_pos[:, idx, :] - root_pos        # (N, 3) world-relative
            foot_b = _quat_rotate_inverse_np(root_quat, rel_w)  # (N, 3) base frame
            foot_parts.append(foot_b)

        amp_obs_np = np.concatenate(
            [jpos_rel, jvel, base_lin_vel, base_ang_vel] + foot_parts, axis=1
        )
        tensor = torch.from_numpy(amp_obs_np).to(self.device)
        self.data = tensor
        return tensor

    def _load_csv(self, path: str) -> torch.Tensor:
        """Load LAFAN1-style CSV and convert to flat AMP feature vectors.

        Column layout (G1, 30 FPS):
          ``[x, y, z, qx, qy, qz, qw,  joint_0 … joint_N-1]``

        Velocities are computed via finite difference at the file's native FPS
        (default 30 Hz).

        .. warning::
            Foot positions require FK and are **not** available from CSV alone.
            They are set to **zero** in the output.  If foot position features
            are important for your discriminator, convert to .npz format first
            or use the AMASS dataset instead.
        """
        data_np = np.loadtxt(path, delimiter=",", dtype=np.float32)  # (N, 36+)
        fps = self._profile.get("csv_fps", 30.0)

        root_pos = data_np[:, :3]   # (N, 3)  x, y, z
        quat_csv = data_np[:, 3:7]  # (N, 4)  format depends on robot

        # Convert quaternion to [w, x, y, z]
        csv_fmt = self._profile.get("csv_quat_format", "xyzw")
        if csv_fmt == "xyzw":
            # input: qx, qy, qz, qw → output: qw, qx, qy, qz
            root_quat = np.concatenate([quat_csv[:, 3:4], quat_csv[:, :3]], axis=1)
        else:
            root_quat = quat_csv  # already [w,x,y,z]

        num_dof = self._profile.get("num_dof", data_np.shape[1] - 7)
        joint_pos = data_np[:, 7: 7 + num_dof]  # (N, J)

        # joint_pos_rel
        if self._default_jpos is not None:
            jpos_rel = joint_pos - self._default_jpos[None, :]
        else:
            jpos_rel = joint_pos

        # velocities via finite difference
        jvel = _finite_diff(joint_pos, fps)                           # (N, J)
        root_lin_vel_w = _finite_diff(root_pos, fps)                  # (N, 3)
        base_lin_vel = _quat_rotate_inverse_np(root_quat, root_lin_vel_w)  # (N, 3)
        base_ang_vel = _ang_vel_from_quats(root_quat, fps)            # (N, 3)

        # foot positions: not available from CSV → zeros
        num_feet = len(self._profile.get("foot_body_indices", [2]))
        foot_zeros = np.zeros((data_np.shape[0], num_feet * 3), dtype=np.float32)
        if num_feet > 0:
            logger.warning(
                "[MotionLoader] CSV format does not contain body positions. "
                "Foot position features are set to zero. "
                "For best results, use the AMASS .npz dataset."
            )

        amp_obs_np = np.concatenate(
            [jpos_rel, jvel, base_lin_vel, base_ang_vel, foot_zeros], axis=1
        )
        tensor = torch.from_numpy(amp_obs_np).to(self.device)
        self.data = tensor
        return tensor

    def _load_directory(self, path: str) -> torch.Tensor:
        """Recursively load all supported motion files from a directory."""
        all_data = []
        supported = (".npy", ".pt", ".pth", ".npz", ".csv")
        for root, _, files in os.walk(path):
            for filename in sorted(files):
                if not filename.lower().endswith(supported):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    chunk = self.load(filepath)
                    all_data.append(chunk)
                except Exception as e:
                    logger.warning(f"[MotionLoader] Skipping {filepath}: {e}")

        if not all_data:
            raise FileNotFoundError(f"No supported motion files found under {path}")

        tensor = torch.cat(all_data, dim=0)
        self.data = tensor
        return tensor

    def sample(self, batch_size: int, history_length: int = 2) -> torch.Tensor:
        """Sample random consecutive-frame clips from the loaded motion data.

        When ``history_length > 1`` the sampler draws pairs (or longer clips) of
        **consecutive** frames from the same motion clip and concatenates them
        along the feature dimension, so the output matches the AMP observation
        that the environment produces with ``ObservationGroupCfg.history_length``
        set to the same value (``flatten_history_dim=True`` is the IsaacLab
        default).

        Args:
            batch_size: Number of clips to sample.
            history_length: Number of consecutive frames per clip.  Must match
                the ``history_length`` set on the AMP observation group in the
                environment config (default 2).

        Returns:
            Sampled clips.
            Shape: ``(batch_size, history_length * obs_dim_per_frame)``
        """
        if self.data is None:
            raise RuntimeError("No motion data loaded. Call load() first.")
        num_frames = self.data.shape[0]
        if history_length <= 1:
            indices = torch.randint(0, num_frames, (batch_size,), device=self.device)
            return self.data[indices]
        # Sample start indices such that start + history_length - 1 < num_frames
        indices = torch.randint(0, num_frames - history_length + 1, (batch_size,), device=self.device)
        # Gather consecutive frames: shape (batch_size, history_length, obs_dim)
        offsets = torch.arange(history_length, device=self.device)  # (history_length,)
        frame_indices = indices.unsqueeze(1) + offsets.unsqueeze(0)  # (batch_size, history_length)
        clips = self.data[frame_indices]  # (batch_size, history_length, obs_dim)
        # Flatten history dim to match env output: (batch_size, history_length * obs_dim)
        return clips.reshape(batch_size, -1)

    @property
    def num_frames(self) -> int:
        """Number of frames in the loaded data."""
        return 0 if self.data is None else self.data.shape[0]

    @property
    def obs_dim(self) -> int:
        """Observation dimension of the loaded data."""
        return 0 if self.data is None else self.data.shape[1]
