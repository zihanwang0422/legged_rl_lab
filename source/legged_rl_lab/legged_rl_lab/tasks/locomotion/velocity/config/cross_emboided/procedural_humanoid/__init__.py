# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Procedural Humanoid Environment Registration.

This module registers Gym environments for training RL agents on
procedurally generated humanoid robots using the metamorphosis framework.
"""

import gymnasium as gym

from . import agents


# Use custom environment class for procedural robots
_PROCEDURAL_ENV_ENTRY_POINT = "legged_rl_lab.tasks.locomotion.velocity.config.cross_emboided.g1go2_mixed.mdp.procedural_obs:ProceduralRobotEnv"


gym.register(
    id="LeggedRLLab-Isaac-Velocity-Flat-Procedural-Humanoid-v0",
    entry_point=_PROCEDURAL_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.procedural_env_cfg:ProceduralHumanoidFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralHumanoidFlatPPORunnerCfg",
    },
)


gym.register(
    id="LeggedRLLab-Isaac-Velocity-Flat-Procedural-Humanoid-Play-v0",
    entry_point=_PROCEDURAL_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.procedural_env_cfg:ProceduralHumanoidFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralHumanoidFlatPPORunnerCfg",
    },
)


gym.register(
    id="LeggedRLLab-Isaac-Velocity-Rough-Procedural-Humanoid-v0",
    entry_point=_PROCEDURAL_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.procedural_env_cfg:ProceduralHumanoidRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralHumanoidRoughPPORunnerCfg",
    },
)


gym.register(
    id="LeggedRLLab-Isaac-Velocity-Rough-Procedural-Humanoid-Play-v0",
    entry_point=_PROCEDURAL_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.procedural_env_cfg:ProceduralHumanoidRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralHumanoidRoughPPORunnerCfg",
    },
)
