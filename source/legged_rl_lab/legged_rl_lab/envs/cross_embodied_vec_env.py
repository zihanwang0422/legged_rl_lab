# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Cross-embodied VecEnv wrapper.

Combines two independently wrapped ``VecEnv`` instances (e.g. G1 and Go2) into
a single unified RSL-RL ``VecEnv``.

Architecture
------------
- All envs from *env_a* (robot A) come first in the batch dimension.
- All envs from *env_b* (robot B) follow immediately after.
- Observations are **zero-padded** to the maximum per-group dimension and
  a 2-dim one-hot robot-type vector is appended:  ``[1, 0]`` = robot A,
  ``[0, 1]`` = robot B.
- The unified action dimension equals the maximum of the two envs' action dims.
  Robot B envs only receive the first ``env_b.num_actions`` dims; remaining dims
  are ignored.

.. warning::
    Isaac Sim supports only a single simulation stage per Python process.
    This wrapper is provided for testing (e.g. with mock envs) and for
    future Isaac Sim releases that support multi-stage execution.  For
    production dual-robot training use ``train_cross_embodied_dual.py``
    (separate processes) or the ``g1go2_mixed`` single-scene env.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

if TYPE_CHECKING:
    pass


class CrossEmbodiedVecEnv(VecEnv):
    """Unified VecEnv combining two independent VecEnv instances.

    Parameters
    ----------
    env_a:
        The first wrapped environment (robot A, e.g. G1).
    env_b:
        The second wrapped environment (robot B, e.g. Go2).
    device:
        Torch device for unified tensors.  Both sub-envs must run on this
        device.
    """

    def __init__(
        self,
        env_a: VecEnv,
        env_b: VecEnv,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()

        self.env_a = env_a
        self.env_b = env_b
        self._device = device

        self._n_a: int = env_a.num_envs
        self._n_b: int = env_b.num_envs

        # --------------- action dims ----------------------------------------
        self._act_dim_a: int = env_a.num_actions
        self._act_dim_b: int = env_b.num_actions
        self._unified_act_dim: int = max(self._act_dim_a, self._act_dim_b)

        # --------------- obs dims (auto-detected after first get_obs call) ---
        self._obs_dim_a: int | None = None
        self._obs_dim_b: int | None = None
        self._unified_obs_dim: int | None = None  # filled lazily

        self._critic_dim_a: int | None = None
        self._critic_dim_b: int | None = None
        self._unified_critic_dim: int | None = None

        # One-hot robot type vectors
        self._type_a = torch.tensor([1.0, 0.0], device=device)  # robot A
        self._type_b = torch.tensor([0.0, 1.0], device=device)  # robot B

        # Max episode length  (use the maximum of both envs)
        self._max_episode_length = max(env_a.max_episode_length, env_b.max_episode_length)

        # A shared config object – expose env_a's cfg as a convenience.
        self.cfg = env_a.cfg

        # Warm up: obtain initial unified observations.
        obs_a = self.env_a.get_observations()
        obs_b = self.env_b.get_observations()
        self._cached_obs = self._combine_obs(obs_a, obs_b)

    # ------------------------------------------------------------------
    # VecEnv abstract property implementations
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._n_a + self._n_b

    @property
    def num_actions(self) -> int:
        return self._unified_act_dim

    @property
    def max_episode_length(self) -> int:
        return self._max_episode_length

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.cat([self.env_a.episode_length_buf, self.env_b.episode_length_buf], dim=0)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env_a.episode_length_buf = value[: self._n_a]
        self.env_b.episode_length_buf = value[self._n_a :]

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # VecEnv abstract method implementations
    # ------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        return self._cached_obs

    def step(self, actions: torch.Tensor):
        """Step both envs and return unified (obs, rewards, dones, extras).

        Parameters
        ----------
        actions:
            Unified action tensor of shape ``(n_a + n_b, unified_act_dim)``.
        """
        actions_a = actions[: self._n_a, : self._act_dim_a]
        actions_b = actions[self._n_a :, : self._act_dim_b]

        obs_a, rew_a, done_a, extras_a = self.env_a.step(actions_a)
        obs_b, rew_b, done_b, extras_b = self.env_b.step(actions_b)

        # Combined observations
        obs = self._combine_obs(obs_a, obs_b)
        self._cached_obs = obs

        # Combined scalar outputs
        rewards = torch.cat([rew_a.to(self._device), rew_b.to(self._device)], dim=0)
        dones = torch.cat([done_a.to(self._device), done_b.to(self._device)], dim=0)

        # Combined extras – keep time_outs mandatory for RSL-RL reward bootstrap
        time_outs_a = extras_a.get("time_outs", torch.zeros(self._n_a, device=self._device, dtype=torch.bool))
        time_outs_b = extras_b.get("time_outs", torch.zeros(self._n_b, device=self._device, dtype=torch.bool))
        extras = {
            "time_outs": torch.cat([time_outs_a.to(self._device), time_outs_b.to(self._device)], dim=0),
        }

        return obs, rewards, dones, extras

    def reset(self):
        obs_a, ex_a = self.env_a.reset()
        obs_b, ex_b = self.env_b.reset()
        obs = self._combine_obs(obs_a, obs_b)
        self._cached_obs = obs
        return obs, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _combine_obs(self, obs_a: TensorDict, obs_b: TensorDict) -> TensorDict:
        """Pad and concatenate observation groups from both envs."""
        combined: dict[str, torch.Tensor] = {}

        obs_groups = set(obs_a.keys()) | set(obs_b.keys())
        for key in obs_groups:
            if key in obs_a and key in obs_b:
                ta = obs_a[key].to(self._device)   # (n_a, dim_a)
                tb = obs_b[key].to(self._device)   # (n_b, dim_b)

                dim_a, dim_b = ta.shape[-1], tb.shape[-1]
                max_dim = max(dim_a, dim_b)

                # Zero-pad to max_dim
                if dim_a < max_dim:
                    ta = torch.nn.functional.pad(ta, (0, max_dim - dim_a))
                if dim_b < max_dim:
                    tb = torch.nn.functional.pad(tb, (0, max_dim - dim_b))

                # Append robot-type one-hot
                onehot_a = self._type_a.expand(self._n_a, -1)  # (n_a, 2)
                onehot_b = self._type_b.expand(self._n_b, -1)  # (n_b, 2)
                ta = torch.cat([ta, onehot_a], dim=-1)          # (n_a, max_dim+2)
                tb = torch.cat([tb, onehot_b], dim=-1)          # (n_b, max_dim+2)

                combined[key] = torch.cat([ta, tb], dim=0)      # (n_a+n_b, max_dim+2)
            elif key in obs_a:
                ta = obs_a[key].to(self._device)
                onehot_a = self._type_a.expand(self._n_a, -1)
                ta = torch.cat([ta, onehot_a], dim=-1)
                # Pad b with zeros for the full width
                tb = torch.zeros(self._n_b, ta.shape[-1], device=self._device)
                combined[key] = torch.cat([ta, tb], dim=0)
            else:
                tb = obs_b[key].to(self._device)
                onehot_b = self._type_b.expand(self._n_b, -1)
                tb = torch.cat([tb, onehot_b], dim=-1)
                ta = torch.zeros(self._n_a, tb.shape[-1], device=self._device)
                combined[key] = torch.cat([ta, tb], dim=0)

        return TensorDict(combined, batch_size=[self._n_a + self._n_b], device=self._device)
