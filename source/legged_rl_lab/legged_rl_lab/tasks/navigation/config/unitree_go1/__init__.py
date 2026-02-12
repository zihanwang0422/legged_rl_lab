# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="LeggedRLLab-Isaac-Navigation-Flat-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:UnitreeGo1NavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1NavigationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_navigation_ppo_cfg.yaml",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-Navigation-Obstacle-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:UnitreeGo1NavigationObstacleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1NavigationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_navigation_ppo_cfg.yaml",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-Navigation-Flat-Unitree-Go1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:UnitreeGo1NavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1NavigationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_navigation_ppo_cfg.yaml",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-Navigation-Obstacle-Unitree-Go1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:UnitreeGo1NavigationObstacleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1NavigationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_navigation_ppo_cfg.yaml",
    },
)
