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

##
# Register Gym environments.
##

_ENTRY_POINT = (
    "legged_rl_lab.tasks.locomotion.cross_emboided"
    ".config.g1go2_mixed.g1go2_flat_env_cfg:CrossEmbodiedG1Go2Env"
)

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Flat-G1Go2-Mixed-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1go2_flat_env_cfg:CrossEmbodiedG1Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrossEmbodiedG1Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Flat-G1Go2-Mixed-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1go2_flat_env_cfg:CrossEmbodiedG1Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrossEmbodiedG1Go2FlatPPORunnerCfg",
    },
)
