# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-PPO algorithm: PPO extended with Adversarial Motion Priors discriminator.

Reference:
    Peng et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Animation."
    ACM Trans. Graph. (SIGGRAPH), 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules.discriminator import AMPDiscriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.amp_storage import AMPReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config


class AMPPPO(PPO):
    """PPO algorithm augmented with an AMP discriminator.

    The discriminator is trained to distinguish between reference motion data (real)
    and policy-generated motions (fake). The policy receives an additional style reward
    derived from the discriminator output.
    """

    discriminator: AMPDiscriminator
    """The AMP discriminator network."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # AMP discriminator parameters
        amp_obs_dim: int = 0,
        amp_discriminator_hidden_dims: list[int] | None = None,
        amp_discriminator_activation: str = "relu",
        amp_learning_rate: float = 1e-3,
        amp_replay_buffer_size: int = 1000000,
        amp_task_reward_lerp: float = 0.5,
        amp_disc_gradient_penalty_coef: float = 5.0,
        amp_disc_logit_reg_coef: float = 0.05,
        amp_disc_weight_decay: float = 0.0001,
        amp_num_learning_epochs: int | None = None,
        amp_num_mini_batches: int | None = None,
        # PPO parameters
        **kwargs,
    ) -> None:
        """Initialize the AMP-PPO algorithm."""
        super().__init__(actor, critic, storage, **kwargs)

        if amp_discriminator_hidden_dims is None:
            amp_discriminator_hidden_dims = [1024, 512]

        self.amp_obs_dim = amp_obs_dim
        self.amp_task_reward_lerp = amp_task_reward_lerp
        self.amp_disc_gradient_penalty_coef = amp_disc_gradient_penalty_coef
        self.amp_disc_logit_reg_coef = amp_disc_logit_reg_coef
        self.amp_num_learning_epochs = amp_num_learning_epochs or self.num_learning_epochs
        self.amp_num_mini_batches = amp_num_mini_batches or self.num_mini_batches

        # Create discriminator
        self.discriminator = AMPDiscriminator(
            input_dim=amp_obs_dim,
            hidden_dims=amp_discriminator_hidden_dims,
            activation=amp_discriminator_activation,
        ).to(self.device)

        # Discriminator optimizer
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=amp_learning_rate,
            weight_decay=amp_disc_weight_decay,
        )

        # AMP replay buffer (stores agent AMP observations for discriminator training)
        self.amp_replay_buffer = AMPReplayBuffer(
            buffer_size=amp_replay_buffer_size,
            obs_dim=amp_obs_dim,
            device=self.device,
        )

        # Reference motion data (set externally via set_reference_data)
        self.amp_reference_data: torch.Tensor | None = None

    def set_reference_data(self, reference_data: torch.Tensor, history_length: int = 2) -> None:
        """Set the reference motion data for the discriminator.

        If ``history_length > 1`` the raw single-frame data (shape ``(N, D)``)
        is pre-processed into consecutive-frame pairs so that each stored row
        has shape ``history_length * D``, matching the AMP observation that the
        environment produces with the same ``history_length`` setting.

        Args:
            reference_data: Raw per-frame AMP observations.
                Shape: ``(N, D)`` for single-frame data, or already
                ``(N, history_length * D)`` if pre-processed externally.
            history_length: Number of consecutive frames to concatenate.
                Must match ``ObservationGroupCfg.history_length`` of the AMP
                observation group (default 2).
        """
        data = reference_data.to(self.device).float()
        if history_length > 1 and data.shape[1] != self.discriminator.input_dim:
            # data is raw single-frame; build consecutive-frame pairs
            # result shape: (N - history_length + 1, history_length * D)
            n = data.shape[0]
            indices = torch.arange(history_length, device=self.device)
            # (N - history_length + 1, history_length, D)
            clips = torch.stack([data[i: n - history_length + 1 + i] for i in range(history_length)], dim=1)
            data = clips.reshape(clips.shape[0], -1)
        self.amp_reference_data = data

    def compute_amp_reward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Compute the AMP style reward from the discriminator.

        The reward is: r_amp = -log(1 - sigmoid(D(s)))
        which encourages the policy to produce motions that the discriminator
        classifies as "real".

        Args:
            amp_obs: AMP observations. Shape: (num_envs, amp_obs_dim)

        Returns:
            Style rewards. Shape: (num_envs,)
        """
        with torch.no_grad():
            disc_logits = self.discriminator(amp_obs)
            prob = torch.sigmoid(disc_logits)
            # Clamp to avoid log(0)
            prob = torch.clamp(prob, 1e-6, 1.0 - 1e-6)
            amp_reward = -torch.log(1.0 - prob).squeeze(-1)
        return amp_reward

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        """Process one environment step, blending task + AMP rewards.

        If AMP observations are provided in extras["amp_obs"], compute style reward
        and blend with task reward.
        """
        # Compute AMP style reward if available
        if "amp_obs" in extras:
            amp_obs = extras["amp_obs"]
            amp_reward = self.compute_amp_reward(amp_obs)
            # Store AMP observations in replay buffer
            self.amp_replay_buffer.insert(amp_obs.detach())
            # Blend task and style rewards
            # r_total = w_task * r_task + (1 - w_task) * r_amp
            blended_rewards = (
                self.amp_task_reward_lerp * rewards
                + (1.0 - self.amp_task_reward_lerp) * amp_reward
            )
            # Store unbounded rewards for logging
            extras["amp_reward_mean"] = amp_reward.mean().item()
            extras["task_reward_mean"] = rewards.mean().item()
        else:
            blended_rewards = rewards

        # Call parent process_env_step with blended rewards
        super().process_env_step(obs, blended_rewards, dones, extras)

    def update(self) -> dict[str, float]:
        """Run PPO update + discriminator update."""
        # Run standard PPO update
        loss_dict = super().update()

        # Run discriminator update
        if self.amp_reference_data is not None and len(self.amp_replay_buffer) > 0:
            disc_loss_dict = self._update_discriminator()
            loss_dict.update(disc_loss_dict)

        return loss_dict

    def _update_discriminator(self) -> dict[str, float]:
        """Train the discriminator on reference (real) vs agent (fake) data."""
        mean_disc_loss = 0.0
        mean_disc_acc_real = 0.0
        mean_disc_acc_fake = 0.0
        mean_grad_penalty = 0.0

        num_updates = 0

        for _ in range(self.amp_num_learning_epochs):
            for _ in range(self.amp_num_mini_batches):
                # Determine batch size
                batch_size = min(
                    512,
                    len(self.amp_replay_buffer),
                    self.amp_reference_data.shape[0],
                )
                if batch_size == 0:
                    continue

                # Sample real data (from reference motions)
                real_indices = torch.randint(
                    0, self.amp_reference_data.shape[0], (batch_size,), device=self.device
                )
                real_data = self.amp_reference_data[real_indices]

                # Sample fake data (from agent replay buffer)
                fake_data = self.amp_replay_buffer.sample(batch_size)

                # Forward pass
                real_logits = self.discriminator(real_data)
                fake_logits = self.discriminator(fake_data)

                # Discriminator loss (least-squares GAN style)
                # Real should map to 1, fake to 0
                disc_loss_real = torch.mean((real_logits - 1.0) ** 2)
                disc_loss_fake = torch.mean((fake_logits + 1.0) ** 2)
                disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)

                # Logit regularization
                logit_reg = self.amp_disc_logit_reg_coef * (
                    torch.mean(real_logits ** 2) + torch.mean(fake_logits ** 2)
                )

                # Gradient penalty on real data
                grad_penalty = self._compute_gradient_penalty(real_data)

                total_disc_loss = disc_loss + logit_reg + self.amp_disc_gradient_penalty_coef * grad_penalty

                # Backward pass
                self.disc_optimizer.zero_grad()
                total_disc_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.disc_optimizer.step()

                # Accuracy
                with torch.no_grad():
                    disc_acc_real = (real_logits > 0).float().mean().item()
                    disc_acc_fake = (fake_logits < 0).float().mean().item()

                mean_disc_loss += disc_loss.item()
                mean_disc_acc_real += disc_acc_real
                mean_disc_acc_fake += disc_acc_fake
                mean_grad_penalty += grad_penalty.item()
                num_updates += 1

        if num_updates > 0:
            mean_disc_loss /= num_updates
            mean_disc_acc_real /= num_updates
            mean_disc_acc_fake /= num_updates
            mean_grad_penalty /= num_updates

        return {
            "amp_disc_loss": mean_disc_loss,
            "amp_disc_acc_real": mean_disc_acc_real,
            "amp_disc_acc_fake": mean_disc_acc_fake,
            "amp_grad_penalty": mean_grad_penalty,
        }

    def _compute_gradient_penalty(self, real_data: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty on real data for discriminator regularization."""
        real_data = real_data.detach().requires_grad_(True)
        disc_output = self.discriminator(real_data)
        grad = torch.autograd.grad(
            outputs=disc_output,
            inputs=real_data,
            grad_outputs=torch.ones_like(disc_output),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_penalty = torch.mean(grad.pow(2).sum(dim=-1))
        return grad_penalty

    def train_mode(self) -> None:
        """Set train mode for all learnable models."""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for all learnable models."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        saved_dict["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        result = super().load(loaded_dict, load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        if "disc_optimizer_state_dict" in loaded_dict:
            self.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
        return result

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        """Construct the AMP-PPO algorithm."""
        # Resolve class callables
        alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Extract AMP config
        amp_cfg = cfg["algorithm"].pop("amp_cfg", {}) or {}
        # Get AMP obs dim from the environment observations
        amp_obs_dim = amp_cfg.pop("amp_obs_dim", 0)
        if amp_obs_dim == 0 and "amp" in obs:
            amp_obs_dim = obs["amp"].shape[-1]

        # Build multi_gpu_cfg argument
        multi_gpu_cfg = cfg.get("multi_gpu", None)

        # Initialize the algorithm
        alg: AMPPPO = alg_class(
            actor, critic, storage,
            amp_obs_dim=amp_obs_dim,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
            **amp_cfg,
            **{k: v for k, v in cfg["algorithm"].items()
               if k not in ("multi_gpu_cfg",)},
        )

        print(f"AMP Discriminator: {alg.discriminator}")
        return alg
