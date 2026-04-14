# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Command generators for cross-embodied locomotion tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from dataclasses import MISSING
from isaaclab.utils import configclass

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(UniformVelocityCommand):
    """Velocity command with small-command threshold and pit-terrain awareness."""

    cfg: "UniformThresholdVelocityCommandCfg"

    def __init__(self, cfg: "UniformThresholdVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids, :2] *= (
            torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def _update_command(self):
        super()._update_command()

        on_pits = is_robot_on_terrain(self._env, "pits")

        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            self._resample_command(torch.where(left_pit_mask)[0])

        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0
            self.vel_command_b[pit_env_ids, 2] = 0.0
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """Command generator that assigns discrete integer commands to environments."""

    cfg: "DiscreteCommandControllerCfg"

    def __init__(self, cfg: "DiscreteCommandControllerCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        self.available_commands = self.cfg.available_commands
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.current_commands = [self.available_commands[0]] * self.num_envs

    def __str__(self) -> str:
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buffer

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """List of available discrete integer commands."""


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
