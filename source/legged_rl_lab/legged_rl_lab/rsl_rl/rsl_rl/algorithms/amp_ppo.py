# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import AMPReplayBuffer, RolloutStorage
from rsl_rl.utils import AMPLoader, Normalizer, resolve_callable, resolve_obs_groups


class Discriminator(nn.Module):
    """TienKung-style AMP discriminator operating on transition pairs."""

    def __init__(
        self,
        input_dim: int,
        amp_reward_coef: float,
        hidden_layer_sizes: list[int],
        device: str | torch.device,
        task_reward_lerp: float = 0.0,
    ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.amp_linear(self.trunk(value))

    def compute_grad_pen(self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad_(True)
        disc = self.forward(expert_data)
        ones = torch.ones_like(disc, device=disc.device)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer: Normalizer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)
            disc = self.forward(torch.cat([state, next_state], dim=-1))
            style_reward = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * torch.square(disc - 1.0), min=0.0)
            reward = style_reward
            if self.task_reward_lerp > 0.0:
                reward = (1.0 - self.task_reward_lerp) * style_reward + self.task_reward_lerp * task_reward.unsqueeze(-1)
            self.train()
        return reward.squeeze(-1), disc, style_reward.squeeze(-1)


class AMPPPO(PPO):
    """PPO + pair-style AMP aligned with TienKung-Lab."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator: Discriminator,
        amp_data: AMPLoader,
        amp_normalizer: Normalizer | None,
        amp_replay_buffer_size: int = 100_000,
        amp_grad_penalty_coef: float = 10.0,
        amp_trunk_weight_decay: float = 1.0e-3,
        amp_head_weight_decay: float = 1.0e-1,
        amp_learning_rate: float | None = None,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.discriminator = discriminator.to(self.device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.amp_storage = AMPReplayBuffer(amp_replay_buffer_size, amp_data.observation_dim, self.device)
        self.amp_grad_penalty_coef = amp_grad_penalty_coef
        self._current_amp_obs: torch.Tensor | None = None
        self.latest_task_rewards: torch.Tensor | None = None
        self.latest_style_rewards: torch.Tensor | None = None
        self.latest_blended_rewards: torch.Tensor | None = None
        self.latest_amp_disc: torch.Tensor | None = None

        self.optimizer.add_param_group(
            {
                "params": self.discriminator.trunk.parameters(),
                "weight_decay": amp_trunk_weight_decay,
                "name": "amp_trunk",
                **({"lr": amp_learning_rate} if amp_learning_rate is not None else {}),
            }
        )
        self.optimizer.add_param_group(
            {
                "params": self.discriminator.amp_linear.parameters(),
                "weight_decay": amp_head_weight_decay,
                "name": "amp_head",
                **({"lr": amp_learning_rate} if amp_learning_rate is not None else {}),
            }
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        if "amp" not in obs.keys():
            raise KeyError(
                "AMP observation group 'amp' not found in env observations. "
                "Ensure the env exposes a single-frame 'amp' observation group."
            )
        self._current_amp_obs = obs["amp"].detach()
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self._current_amp_obs is None:
            raise RuntimeError("AMP transition state was not recorded before env.step().")

        next_amp_obs = extras.get("amp_obs")
        if next_amp_obs is None:
            if "amp" not in obs.keys():
                raise KeyError("Neither extras['amp_obs'] nor obs['amp'] is available for AMP training.")
            next_amp_obs = obs["amp"]
        next_amp_obs = next_amp_obs.to(self.device)

        task_rewards = rewards.to(self.device)
        rewards, disc, style_rewards = self.discriminator.predict_amp_reward(
            self._current_amp_obs, next_amp_obs, rewards.to(self.device), normalizer=self.amp_normalizer
        )
        self.latest_task_rewards = task_rewards.detach()
        self.latest_style_rewards = style_rewards.detach()
        self.latest_blended_rewards = rewards.detach()
        self.latest_amp_disc = disc.detach()
        self.amp_storage.insert(self._current_amp_obs, next_amp_obs)
        super().process_env_step(obs, rewards, dones, extras)
        self._current_amp_obs = None

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches, mini_batch_size
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches, mini_batch_size
        )

        for batch, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"], obs=batch.observations, actions=batch.actions
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            log_ratio = actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
            log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
            ratio = torch.exp(log_ratio)
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions_batch = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                symmetry_loss = torch.nn.MSELoss()(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d, device=self.device))
            policy_loss = torch.nn.MSELoss()(policy_d, -torch.ones_like(policy_d, device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, lambda_=self.amp_grad_penalty_coef
            )
            loss += amp_loss + grad_pen_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.detach().cpu().numpy())
                self.amp_normalizer.update(policy_next_state.detach().cpu().numpy())
                self.amp_normalizer.update(expert_state.detach().cpu().numpy())
                self.amp_normalizer.update(expert_next_state.detach().cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        saved_dict["amp_normalizer"] = self.amp_normalizer
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=False)
        if "amp_normalizer" in loaded_dict:
            self.amp_normalizer = loaded_dict["amp_normalizer"]
        return load_iteration

    def broadcast_parameters(self) -> None:
        model_params = [self.actor.state_dict(), self.critic.state_dict(), self.discriminator.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.discriminator.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    def reduce_parameters(self) -> None:
        all_params = chain(self.actor.parameters(), self.critic.parameters(), self.discriminator.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        amp_cfg = cfg["algorithm"].pop("amp_cfg", {}) or {}
        amp_motion_files = getattr(env.cfg, "amp_motion_files", "")
        robot_type = getattr(env.cfg, "robot_type", "g1")
        raw_env = getattr(env, "unwrapped_env", env)
        step_dt = getattr(raw_env, "step_dt", None)
        if step_dt is None:
            sim_cfg = getattr(env.cfg, "sim", None)
            decimation = getattr(env.cfg, "decimation", 1)
            step_dt = getattr(sim_cfg, "dt", 1.0 / 30.0) * decimation

        amp_data = AMPLoader(
            device=device,
            time_between_frames=step_dt,
            motion_files=amp_motion_files,
            robot=robot_type,
            preload_transitions=True,
            num_preload_transitions=amp_cfg.get("amp_num_preload_transitions", 200_000),
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discr_hidden_dims = amp_cfg.get("amp_discr_hidden_dims", amp_cfg.get("amp_discriminator_hidden_dims", [1024, 512, 256]))
        discriminator = Discriminator(
            input_dim=amp_data.observation_dim * 2,
            amp_reward_coef=amp_cfg.get("amp_reward_coef", 0.3),
            hidden_layer_sizes=discr_hidden_dims,
            device=device,
            task_reward_lerp=amp_cfg.get("amp_task_reward_lerp", 0.7),
        ).to(device)

        return alg_class(
            actor=actor,
            critic=critic,
            storage=storage,
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            amp_replay_buffer_size=amp_cfg.get("amp_replay_buffer_size", 200_000),
            amp_grad_penalty_coef=amp_cfg.get("amp_disc_gradient_penalty_coef", 10.0),
            amp_trunk_weight_decay=amp_cfg.get("amp_disc_weight_decay", 1.0e-3),
            amp_head_weight_decay=amp_cfg.get("amp_disc_head_weight_decay", 1.0e-1),
            amp_learning_rate=amp_cfg.get("amp_learning_rate"),
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
