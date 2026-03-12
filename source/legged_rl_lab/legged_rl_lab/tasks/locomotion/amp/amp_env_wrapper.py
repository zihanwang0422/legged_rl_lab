# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP-aware VecEnv wrapper for rsl_rl.

This wrapper extends the standard RslRlVecEnvWrapper to expose AMP observations
in the extras dict during env.step(). The AMP-PPO algorithm uses these to compute
discriminator-based style rewards.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class AmpRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Wrapper that extracts AMP observations and passes them through extras.

    The AMP observation group (defined in the environment config) is extracted
    during each step and placed into extras["amp_obs"] for the AMP-PPO algorithm.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        clip_actions: float | None = None,
        amp_obs_group: str = "amp",
    ):
        """Initialize the AMP wrapper.

        Args:
            env: The environment to wrap.
            clip_actions: The clipping value for actions.
            amp_obs_group: The name of the observation group containing AMP obs.
        """
        super().__init__(env, clip_actions=clip_actions)
        self.amp_obs_group = amp_obs_group

    def get_observations(self) -> TensorDict:
        """Returns the current observations including AMP observations."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the environment and extract AMP observations.

        Returns:
            obs: Observations (TensorDict with all groups including "amp").
            rew: Rewards.
            dones: Done flags.
            extras: Contains "amp_obs" key with flattened AMP observations.
        """
        # Clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # Step the environment
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # Compute dones
        dones = (terminated | truncated).to(dtype=torch.long)

        # Handle time outs
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # Extract AMP observations from the observation dict
        if self.amp_obs_group in obs_dict:
            amp_obs = obs_dict[self.amp_obs_group]
            if isinstance(amp_obs, dict):
                # If it's a dict of terms, concatenate them
                amp_obs = torch.cat(list(amp_obs.values()), dim=-1)
            extras["amp_obs"] = amp_obs

        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras
