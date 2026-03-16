# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-specific RSL-RL VecEnv wrapper.

Extends the standard ``RslRlVecEnvWrapper`` to pass AMP observation data
through ``extras["amp_obs"]`` for the AMP-PPO algorithm.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv


class AmpRslRlVecEnvWrapper(VecEnv):
    """Wraps a Gymnasium environment to serve as an RSL-RL VecEnv with AMP support.

    Extracts AMP observations from the ``amp_obs_group`` observation group and
    passes them through ``extras["amp_obs"]`` for the training script.

    This wrapper is robot-agnostic and works with any environment that has an
    AMP observation group configured.

    Following the standard RSL-RL wrapper pattern, ``__init__`` calls
    ``env.reset()`` to obtain the initial observations, which are cached and
    returned by ``get_observations()``.  The cache is updated after every
    ``step()`` / ``reset()`` call.
    """

    def __init__(
        self,
        env: gym.Env,
        clip_actions: float | None = None,
        amp_obs_group: str = "amp",
    ) -> None:
        super().__init__()
        self.env = env
        self.unwrapped_env = env.unwrapped

        # Action clipping
        self._clip_actions = clip_actions

        # AMP obs group name
        self._amp_obs_group = amp_obs_group

        # Get environment info
        self.num_envs = self.unwrapped_env.num_envs
        self.num_actions = self.unwrapped_env.action_manager.action.shape[1]
        self.max_episode_length = int(
            self.unwrapped_env.max_episode_length
            if hasattr(self.unwrapped_env, "max_episode_length")
            else 1000
        )
        self.device = self.unwrapped_env.device

        # Store cfg reference
        self.cfg = self.unwrapped_env.cfg

        # Call reset() to get initial obs — this is the canonical pattern for
        # RSL-RL wrappers and avoids relying on obs_buf being pre-populated.
        obs_dict, _ = self.env.reset()
        self._obs = self._build_obs_tensordict(obs_dict)

    # ------------------------------------------------------------------
    # episode_length_buf property — delegates to underlying env so that
    # the RSL-RL runner's episode-length randomisation actually takes
    # effect on the IsaacLab env, matching `RslRlVecEnvWrapper` behaviour.
    # ------------------------------------------------------------------

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped_env.episode_length_buf = value

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def get_observations(self):
        """Return the cached observation TensorDict."""
        return self._obs

    def step(self, actions: torch.Tensor):
        """Step the environment.

        Returns:
            Tuple of (obs, rewards, dones, extras).
        """
        if self._clip_actions is not None:
            actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs_dict, rewards, terminated, time_outs, extras = self.env.step(actions)

        # Combine terminated and time_outs into dones
        dones = (terminated | time_outs).to(dtype=torch.long)

        # Pass time_outs through extras for reward bootstrapping
        extras["time_outs"] = time_outs

        # Build obs TensorDict and cache
        obs = self._build_obs_tensordict(obs_dict)
        self._obs = obs

        # Pre-reset AMP obs are already in extras["amp_obs"] from AMPManagerBasedRLEnv

        return obs, rewards, dones, extras

    def reset(self):
        """Reset the environment and update the cached observations."""
        obs_dict, extras = self.env.reset()
        obs = self._build_obs_tensordict(obs_dict)
        self._obs = obs
        return obs, extras

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs_tensordict(self, obs_dict):
        """Convert an IsaacLab obs dict to a TensorDict for RSL-RL.

        Handles plain ``dict``, ``TensorDict``, and any mapping whose values
        are either ``torch.Tensor`` instances or tensor-like objects with a
        ``.shape`` attribute (e.g. nested TensorDicts with concat dim).
        """
        from tensordict import TensorDict

        result = {}
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                result[key] = value
            elif hasattr(value, "shape") and hasattr(value, "to"):
                # tensor-like (e.g. TensorDict with a single tensor inside)
                result[key] = value
            elif hasattr(value, "items"):
                # nested mapping — flatten one level
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        result[f"{key}/{subkey}"] = subvalue

        if not result:
            raise RuntimeError(
                f"No tensors found in obs_dict. "
                f"Keys: {list(obs_dict.keys())}. "
                f"Value types: {[type(v).__name__ for v in obs_dict.values()]}"
            )

        if "policy" in result:
            batch_size = result["policy"].shape[0]
        else:
            batch_size = next(iter(result.values())).shape[0]

        return TensorDict(result, batch_size=[batch_size], device=self.device)
