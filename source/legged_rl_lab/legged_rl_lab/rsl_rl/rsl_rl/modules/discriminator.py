# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP Discriminator network.

Distinguishes between expert (reference motion) and policy-generated transitions.
Input: concatenation of (state, next_state) AMP observation pairs.
Output: single logit score.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .normalization import EmpiricalNormalization
from .mlp import MLP
from rsl_rl.utils import resolve_nn_activation


class AMPDiscriminator(nn.Module):
    """Discriminator for Adversarial Motion Priors (AMP).

    Takes concatenated (current_amp_obs, next_amp_obs) as input and outputs
    a logit indicating whether the transition is from the expert dataset.

    Supports:
    - BCEWithLogits loss with R1 gradient penalty
    - Optional empirical normalization of AMP observations
    """

    def __init__(
        self,
        amp_obs_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (1024, 512),
        activation: str = "relu",
        reward_scale: float = 1.0,
        gradient_penalty_coef: float = 5.0,
        logit_reg_coef: float = 0.05,
        weight_decay: float = 1e-4,
        obs_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.amp_obs_dim = amp_obs_dim
        self.input_dim = amp_obs_dim * 2  # concatenation of (s, s')
        self.reward_scale = reward_scale
        self.gradient_penalty_coef = gradient_penalty_coef
        self.logit_reg_coef = logit_reg_coef
        self.weight_decay = weight_decay

        # Trunk MLP (without final activation)
        self.trunk = MLP(
            input_dim=self.input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=list(hidden_dims[:-1]),
            activation=activation,
            last_activation=activation,
        )
        # Linear head
        self.head = nn.Linear(hidden_dims[-1], 1)

        # Observation normalizer (normalizes each half independently)
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(amp_obs_dim)
        else:
            self.obs_normalizer = None

    def forward(self, amp_obs_pair: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            amp_obs_pair: Concatenated (state, next_state) of shape ``(B, input_dim)``.

        Returns:
            Logit of shape ``(B, 1)``.
        """
        if self.obs_normalizer is not None:
            # Normalize each half independently.
            # Validate input dimension before splitting to give a clear error.
            pair_dim = amp_obs_pair.shape[-1]
            if pair_dim != self.input_dim:
                raise ValueError(
                    f"AMPDiscriminator.forward: expected input dim {self.input_dim} "
                    f"(= 2 × amp_obs_dim {self.amp_obs_dim}), but got {pair_dim}. "
                    f"Check that env amp obs dim ({pair_dim // 2} per step) matches "
                    f"motion_loader obs dim ({self.amp_obs_dim // 2} per frame). "
                    f"Ensure foot_positions is consistently enabled/disabled in both."
                )
            s, s_next = amp_obs_pair.chunk(2, dim=-1)
            s = self.obs_normalizer(s)
            s_next = self.obs_normalizer(s_next)
            amp_obs_pair = torch.cat([s, s_next], dim=-1)

        features = self.trunk(amp_obs_pair)
        return self.head(features)

    def predict_reward(self, amp_obs: torch.Tensor, amp_obs_next: torch.Tensor) -> torch.Tensor:
        """Compute style reward from a transition pair (LSGAN formulation).

        LSGAN reward: r = max(0, 1 - 0.25*(D(s,s') - 1)^2)
        where D = tanh(logit) ∈ (-1, 1).
        - When D=+1 (expert-like): r = reward_scale × 1.0  (maximum)
        - When D= 0 (boundary):   r = reward_scale × 0.75
        - When D=-1 (policy-like): r = reward_scale × 0.0  (minimum)

        The bounded output prevents reward divergence that occurs with
        softplus(logit) as the discriminator becomes more confident.

        Args:
            amp_obs: Current AMP observation ``(B, amp_obs_dim)``.
            amp_obs_next: Next AMP observation ``(B, amp_obs_dim)``.

        Returns:
            Style reward ``(B, 1)``.
        """
        with torch.no_grad():
            pair = torch.cat([amp_obs, amp_obs_next], dim=-1)
            d_tanh = torch.tanh(self.forward(pair))  # (B,1) in (-1, 1)
            reward = self.reward_scale * torch.clamp(1.0 - 0.25 * (d_tanh - 1.0) ** 2, min=0.0)
        return reward

    def compute_loss(
        self,
        policy_pair: torch.Tensor,
        expert_pair: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute discriminator loss with gradient penalty.

        Args:
            policy_pair: Policy-generated (s, s') pairs ``(B, input_dim)``.
            expert_pair: Expert (s, s') pairs ``(B, input_dim)``.

        Returns:
            Tuple of (amp_loss, grad_pen_loss, logit_reg_loss).
        """
        # Forward pass on both
        all_pairs = torch.cat([policy_pair, expert_pair], dim=0)
        all_logits = self.forward(all_pairs)
        policy_logits, expert_logits = all_logits.split(policy_pair.shape[0], dim=0)

        # LSGAN loss: expert → 1, policy → -1  (outputs in raw logit space)
        # E[(D(expert) - 1)^2] + E[(D(policy) + 1)^2]  using tanh squashing
        # Equivalent formulation via logits: MSE against ±1 targets on tanh(logit)
        policy_out = torch.tanh(policy_logits)   # mapped to (-1, 1)
        expert_out = torch.tanh(expert_logits)   # mapped to (-1, 1)
        amp_loss = 0.5 * (
            torch.mean((expert_out - 1.0) ** 2)
            + torch.mean((policy_out + 1.0) ** 2)
        )

        # R1 gradient penalty on expert data
        grad_pen_loss = self._compute_gradient_penalty(expert_pair)

        # Logit regularization
        logit_reg_loss = self.logit_reg_coef * torch.mean(expert_logits**2 + policy_logits**2)

        return amp_loss, grad_pen_loss, logit_reg_loss

    def _compute_gradient_penalty(self, expert_pair: torch.Tensor) -> torch.Tensor:
        """R1 gradient penalty on expert data.

        Computes: lambda/2 * E[||grad D(x_expert)||^2]
        """
        expert_pair = expert_pair.detach().requires_grad_(True)

        # Normalize if needed
        if self.obs_normalizer is not None:
            s, s_next = expert_pair.chunk(2, dim=-1)
            s_norm = self.obs_normalizer(s)
            s_next_norm = self.obs_normalizer(s_next)
            pair_norm = torch.cat([s_norm, s_next_norm], dim=-1)
        else:
            pair_norm = expert_pair

        features = self.trunk(pair_norm)
        logits = self.head(features)

        grad = torch.autograd.grad(
            outputs=logits,
            inputs=expert_pair,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_penalty = 0.5 * self.gradient_penalty_coef * torch.mean(grad.pow(2))
        return grad_penalty

    def update_normalizer(self, amp_obs: torch.Tensor) -> None:
        """Update the observation normalizer with new data."""
        if self.obs_normalizer is not None:
            self.obs_normalizer.update(amp_obs)
