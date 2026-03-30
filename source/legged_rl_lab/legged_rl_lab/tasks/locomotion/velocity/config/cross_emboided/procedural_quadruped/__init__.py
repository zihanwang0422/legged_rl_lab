# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Procedural Quadruped Environment Registration.

This module registers Gym environments for training RL agents on
procedurally generated quadruped robots using the metamorphosis framework.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Use custom environment class for procedural robots
_PROCEDURAL_ENV_ENTRY_POINT = "legged_rl_lab.tasks.locomotion.velocity.config.cross_emboided.g1go2_mixed.mdp.procedural_obs:ProceduralRobotEnv"

# Flat terrain environment
gym.register(
    id="LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-v0",
    entry_point=_PROCEDURAL_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.procedural_env_cfg:ProceduralQuadrupedFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralQuadrupedFlatPPORunnerCfg",
    },
)

# Flat terrain play environment
gym.register(
    id="LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-Play-v0",
    entry_point=_PROCEDURAL_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.procedural_env_cfg:ProceduralQuadrupedFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralQuadrupedFlatPPORunnerCfg",
    },
)

# Rough terrain environment
gym.register(
    id="LeggedRLLab-Isaac-Velocity-Rough-Procedural-Quadruped-v0",
    entry_point=_PROCEDURAL_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.procedural_env_cfg:ProceduralQuadrupedRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralQuadrupedRoughPPORunnerCfg",
    },
)

# Rough terrain play environment
gym.register(
    id="LeggedRLLab-Isaac-Velocity-Rough-Procedural-Quadruped-Play-v0",
    entry_point=_PROCEDURAL_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.procedural_env_cfg:ProceduralQuadrupedRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralQuadrupedRoughPPORunnerCfg",
    },
)
