# Copyright (c) 2026, The Legged Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class AttentionTerrainModel(nn.Module):
    """AME-style terrain attention model compatible with the split actor/critic PPO stack."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        map_scan_dim: tuple[int, int, int] | list[int] = (33, 21, 3),
        mha_dim: int = 64,
        num_heads: int = 16,
        cnn_downsample: bool = True,
        attach_global: bool = False,
    ) -> None:
        super().__init__()

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.map_scan_dim = tuple(map_scan_dim)
        self.length, self.width, self.coord_dim = self.map_scan_dim
        self.map_scan_size = self.length * self.width * self.coord_dim
        self.proprio_dim = self.obs_dim - self.map_scan_size
        if self.proprio_dim <= 0:
            raise ValueError(
                f"AttentionTerrainModel expects obs_set '{obs_set}' to end with a flattened map of "
                f"{self.map_scan_size} values from map_scan_dim={self.map_scan_dim}, got obs_dim={self.obs_dim}."
            )

        self.mha_dim = mha_dim
        self.num_heads = num_heads
        self.cnn_downsample = cnn_downsample
        self.attach_global = attach_global
        self.last_attention_weights: torch.Tensor | None = None

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.proprio_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        self._build_terrain_encoder()

        head_input_dim = self.proprio_dim + self.mha_dim
        if attach_global:
            head_input_dim += self.mha_dim

        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.mlp = MLP(head_input_dim, mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        flat_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        return self._encode_terrain(flat_obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def as_jit(self) -> nn.Module:
        return _TorchAttentionTerrainModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxAttentionTerrainModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            flat_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(flat_obs[:, : self.proprio_dim])  # type: ignore[union-attr]

    def _build_terrain_encoder(self) -> None:
        if self.cnn_downsample:
            self.map_cnn = nn.Sequential(
                nn.Conv2d(self.coord_dim, 16, kernel_size=5, padding=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, self.mha_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(self.mha_dim),
            )
        else:
            self.map_cnn = nn.Sequential(
                nn.Conv2d(self.coord_dim, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, self.mha_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(self.mha_dim),
            )

        self.proprio_embedding = nn.Linear(self.proprio_dim, self.mha_dim)
        if self.attach_global:
            self.global_encoder = MLP(self.mha_dim, self.mha_dim, [256, 128], "elu")
            self.query_projector = nn.Linear(self.mha_dim * 2, self.mha_dim)
        else:
            self.global_encoder = None
            self.query_projector = None
        self.mha = nn.MultiheadAttention(embed_dim=self.mha_dim, num_heads=self.num_heads, batch_first=True)

    def _encode_terrain(self, flat_obs: torch.Tensor) -> torch.Tensor:
        if flat_obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected flat obs dim {self.obs_dim}, got {flat_obs.shape[-1]}.")

        proprio_obs = flat_obs[:, : self.proprio_dim]
        map_scan = flat_obs[:, self.proprio_dim :].reshape(
            -1, self.width, self.length, self.coord_dim
        )

        proprio_obs = self.obs_normalizer(proprio_obs)
        map_features = map_scan.permute(0, 3, 1, 2)
        cnn_features = self.map_cnn(map_features)
        local_features = cnn_features.permute(0, 2, 3, 1).reshape(flat_obs.shape[0], -1, self.mha_dim)

        query = self.proprio_embedding(proprio_obs)
        if self.attach_global:
            global_features = self.global_encoder(local_features)  # type: ignore[operator]
            global_features_max = torch.max(global_features, dim=1).values
            query = self.query_projector(torch.cat([global_features_max, query], dim=-1))  # type: ignore[operator]
        else:
            global_features_max = None

        attention_out, attention_weights = self.mha(
            query=query.unsqueeze(1),
            key=local_features,
            value=local_features,
        )
        self.last_attention_weights = attention_weights

        encoded = torch.cat([attention_out.squeeze(1), proprio_obs], dim=-1)
        if global_features_max is not None:
            encoded = torch.cat([global_features_max, encoded], dim=-1)
        return encoded

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"AttentionTerrainModel only supports 1D flattened observations, got "
                    f"{obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchAttentionTerrainModel(nn.Module):
    def __init__(self, model: AttentionTerrainModel) -> None:
        super().__init__()
        self.proprio_dim = model.proprio_dim
        self.length = model.length
        self.width = model.width
        self.coord_dim = model.coord_dim
        self.mha_dim = model.mha_dim
        self.attach_global = model.attach_global
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.map_cnn = copy.deepcopy(model.map_cnn)
        self.proprio_embedding = copy.deepcopy(model.proprio_embedding)
        self.mha = copy.deepcopy(model.mha)
        self.global_encoder = copy.deepcopy(model.global_encoder) if model.global_encoder is not None else nn.Identity()
        self.query_projector = (
            copy.deepcopy(model.query_projector) if model.query_projector is not None else nn.Identity()
        )
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proprio_obs = self.obs_normalizer(x[:, : self.proprio_dim])
        map_scan = x[:, self.proprio_dim :].reshape(-1, self.width, self.length, self.coord_dim)
        map_features = map_scan.permute(0, 3, 1, 2)
        cnn_features = self.map_cnn(map_features)
        local_features = cnn_features.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.mha_dim)
        query = self.proprio_embedding(proprio_obs)
        if self.attach_global:
            global_features = self.global_encoder(local_features)
            global_features_max = torch.max(global_features, dim=1).values
            query = self.query_projector(torch.cat([global_features_max, query], dim=-1))
        else:
            global_features_max = None
        attention_out, _ = self.mha(query=query.unsqueeze(1), key=local_features, value=local_features)
        encoded = torch.cat([attention_out.squeeze(1), proprio_obs], dim=-1)
        if global_features_max is not None:
            encoded = torch.cat([global_features_max, encoded], dim=-1)
        out = self.mlp(encoded)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxAttentionTerrainModel(_TorchAttentionTerrainModel):
    is_recurrent: bool = False

    def __init__(self, model: AttentionTerrainModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
