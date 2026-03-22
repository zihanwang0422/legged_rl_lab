# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Cross-embodied G1 + Go2 mixed-scene locomotion environments.

Both Unitree G1 (29 DOF) and Unitree Go2 (12 DOF) are spawned in a single
Isaac Lab scene.  The first half of env instances run G1; the second half
run Go2.  A shared network (98-dim actor obs, 101-dim critic obs, 29-dim
actions) is trained across both embodiments simultaneously.
"""

import gymnasium as gym

from . import agents
from .g1go2_flat_env_cfg import CrossEmbodiedG1Go2Env, CrossEmbodiedG1Go2FlatEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="LeggedRLLab-Isaac-Velocity-Flat-G1Go2-Mixed-v0",
    # Use the custom env class so that robot-type bookkeeping and parking logic
    # are executed during every reset.
    entry_point=(
        "legged_rl_lab.tasks.locomotion.velocity.config.cross_emboided"
        ".g1go2_mixed.g1go2_flat_env_cfg:CrossEmbodiedG1Go2Env"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1go2_flat_env_cfg:CrossEmbodiedG1Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrossEmbodiedG1Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-Velocity-Flat-G1Go2-Mixed-Play-v0",
    entry_point=(
        "legged_rl_lab.tasks.locomotion.velocity.config.cross_emboided"
        ".g1go2_mixed.g1go2_flat_env_cfg:CrossEmbodiedG1Go2Env"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1go2_flat_env_cfg:CrossEmbodiedG1Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrossEmbodiedG1Go2FlatPPORunnerCfg",
    },
)
