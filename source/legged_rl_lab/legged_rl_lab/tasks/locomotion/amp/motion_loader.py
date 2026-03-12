# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Motion data loader for AMP reference motions.

Supports loading motion clips from .npy or .pt files.
Each motion clip should contain per-frame AMP observation features.

Expected data format:
    - .npy: numpy array of shape (num_frames, obs_dim) or a dict with key "amp_obs"
    - .pt: torch tensor of shape (num_frames, obs_dim) or a dict with key "amp_obs"
    - directory: loads all .npy/.pt files and concatenates them

The AMP observation features typically include:
    - joint positions (relative to default)
    - joint velocities
    - foot positions in base frame
    - base angular velocity
    - base linear velocity (optional)
"""

from __future__ import annotations

import os

import numpy as np
import torch


class MotionLoader:
    """Loads and stores reference motion data for AMP training."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.data: torch.Tensor | None = None

    def load(self, path: str) -> torch.Tensor:
        """Load motion data from a file or directory.

        Args:
            path: Path to a .npy/.pt file or a directory containing such files.

        Returns:
            Loaded motion data tensor. Shape: (total_frames, obs_dim)
        """
        if os.path.isdir(path):
            return self._load_directory(path)
        elif path.endswith(".npy"):
            return self._load_npy(path)
        elif path.endswith(".pt") or path.endswith(".pth"):
            return self._load_pt(path)
        else:
            raise ValueError(f"Unsupported file format: {path}. Use .npy, .pt, or a directory.")

    def _load_npy(self, path: str) -> torch.Tensor:
        """Load from numpy file."""
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            # Dictionary saved as numpy object
            data = data.item()
        if isinstance(data, dict):
            if "amp_obs" in data:
                data = data["amp_obs"]
            else:
                raise KeyError(f"Expected key 'amp_obs' in dict, got keys: {list(data.keys())}")
        tensor = torch.from_numpy(np.array(data, dtype=np.float32)).to(self.device)
        self.data = tensor
        return tensor

    def _load_pt(self, path: str) -> torch.Tensor:
        """Load from PyTorch file."""
        data = torch.load(path, weights_only=True, map_location=self.device)
        if isinstance(data, dict):
            if "amp_obs" in data:
                data = data["amp_obs"]
            else:
                raise KeyError(f"Expected key 'amp_obs' in dict, got keys: {list(data.keys())}")
        tensor = data.float().to(self.device)
        self.data = tensor
        return tensor

    def _load_directory(self, path: str) -> torch.Tensor:
        """Load all motion files from a directory and concatenate."""
        all_data = []
        for filename in sorted(os.listdir(path)):
            filepath = os.path.join(path, filename)
            if filename.endswith(".npy"):
                all_data.append(self._load_npy(filepath))
            elif filename.endswith((".pt", ".pth")):
                all_data.append(self._load_pt(filepath))

        if not all_data:
            raise FileNotFoundError(f"No .npy or .pt files found in {path}")

        tensor = torch.cat(all_data, dim=0)
        self.data = tensor
        return tensor

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample random frames from the loaded motion data.

        Args:
            batch_size: Number of frames to sample.

        Returns:
            Sampled frames. Shape: (batch_size, obs_dim)
        """
        if self.data is None:
            raise RuntimeError("No motion data loaded. Call load() first.")
        indices = torch.randint(0, self.data.shape[0], (batch_size,), device=self.device)
        return self.data[indices]

    @property
    def num_frames(self) -> int:
        """Number of frames in the loaded data."""
        return 0 if self.data is None else self.data.shape[0]

    @property
    def obs_dim(self) -> int:
        """Observation dimension of the loaded data."""
        return 0 if self.data is None else self.data.shape[1]
