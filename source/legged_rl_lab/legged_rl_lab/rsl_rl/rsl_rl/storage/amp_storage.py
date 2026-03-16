# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP replay buffer for storing and sampling AMP observations."""

from __future__ import annotations

import torch


class AMPReplayBuffer:
    """Fixed-size replay buffer for AMP agent observations.

    Stores AMP observations from the policy rollouts to serve as "fake" samples for the
    discriminator. Reference motion data ("real" samples) are managed separately.
    """

    def __init__(self, buffer_size: int, obs_dim: int, device: str = "cpu") -> None:
        self.buffer_size = buffer_size
        self.device = device
        self.obs_dim = obs_dim

        self.buffer = torch.zeros(buffer_size, obs_dim, device=device)
        self.insert_idx = 0
        self.num_stored = 0

    def insert(self, amp_obs: torch.Tensor) -> None:
        """Insert new AMP observations into the buffer.

        Args:
            amp_obs: AMP observations to insert. Shape: (N, obs_dim)
        """
        n = amp_obs.shape[0]
        if n == 0:
            return

        # Wrap-around insertion
        if self.insert_idx + n <= self.buffer_size:
            self.buffer[self.insert_idx : self.insert_idx + n] = amp_obs
        else:
            overflow = (self.insert_idx + n) - self.buffer_size
            first_part = n - overflow
            self.buffer[self.insert_idx : self.insert_idx + first_part] = amp_obs[:first_part]
            self.buffer[:overflow] = amp_obs[first_part:]

        self.insert_idx = (self.insert_idx + n) % self.buffer_size
        self.num_stored = min(self.num_stored + n, self.buffer_size)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample a batch of AMP observations from the buffer.

        Args:
            batch_size: Number of samples to draw.

        Returns:
            Sampled AMP observations. Shape: (batch_size, obs_dim)
        """
        indices = torch.randint(0, self.num_stored, (batch_size,), device=self.device)
        return self.buffer[indices]

    def __len__(self) -> int:
        return self.num_stored
