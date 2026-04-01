# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments for AMP.
##

gym.register(
    id="LeggedRLLab-Isaac-AMP-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_rough_env_cfg:UnitreeGo2AMPFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:UnitreeGo2AMPFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-AMP-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_rough_env_cfg:UnitreeGo2AMPRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:UnitreeGo2AMPRoughPPORunnerCfg",
    },
)
