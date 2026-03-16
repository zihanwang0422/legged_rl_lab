# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP Discriminator network for Adversarial Motion Priors."""

from __future__ import annotations

import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    """Discriminator network for AMP (Adversarial Motion Priors).

    The discriminator takes AMP observations (current + next frame) and outputs a score
    indicating how "real" the motion looks compared to reference motion data.

    Reference:
        Peng et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Animation."
        ACM Trans. Graph. (SIGGRAPH), 2021.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [1024, 512],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Build MLP layers
        activation_fn = self._get_activation(activation)
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_fn)
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        Args:
            amp_obs: AMP observations. Shape: (batch_size, input_dim)

        Returns:
            Discriminator logits (unnormalized scores). Shape: (batch_size, 1)
        """
        return self.net(amp_obs)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "selu": nn.SELU(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        return activations[name]
