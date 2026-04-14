# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Procedural Mixed (Humanoid + Quadruped) Environment Registration.

Registers Gym environments for training a single cross-embodied policy on
a heterogeneous mix of procedurally generated bipeds and quadrupeds.

Encoder variants
----------------
Each terrain type ships three encoder back-ends:
  * ``-Mask-``        : plain flat vector (current approach)
  * ``-Transformer-`` : joint-token attention encoder
  * ``-GCN-``         : graph convolutional encoder
"""

import gymnasium as gym

from . import agents

_MIXED_ENV_ENTRY_POINT = (
    "legged_rl_lab.tasks.locomotion.cross_emboided.mdp.cross_procedural_mdp:ProceduralMixedRobotEnv"
)

# ── Flat – Mask (default) ─────────────────────────────────────────────────

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-Play-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedFlatPPORunnerCfg",
    },
)

# ── Flat – Transformer ────────────────────────────────────────────────────

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-Transformer-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedFlatTransformerPPORunnerCfg",
    },
)

# ── Flat – GCN ────────────────────────────────────────────────────────────

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-GCN-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedFlatGCNPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Flat-Procedural-Mixed-GCN-Play-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedFlatGCNPPORunnerCfg",
    },
)

# ── Rough – Mask (default) ────────────────────────────────────────────────

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-Play-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedRoughPPORunnerCfg",
    },
)

# ── Rough – Transformer ───────────────────────────────────────────────────

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-Transformer-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedRoughTransformerPPORunnerCfg",
    },
)

# ── Rough – GCN ───────────────────────────────────────────────────────────

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-GCN-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedRoughGCNPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-CrossEmboided-Rough-Procedural-Mixed-GCN-Play-v0",
    entry_point=_MIXED_ENV_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:ProceduralMixedRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ProceduralMixedRoughGCNPPORunnerCfg",
    },
)
