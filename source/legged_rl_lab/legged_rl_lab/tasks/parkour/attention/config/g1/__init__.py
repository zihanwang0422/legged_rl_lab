import gymnasium as gym

from . import agents


gym.register(
    id="LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_attention_env_cfg:G1AttentionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AttentionPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-Parkour-Attention-Unitree-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_attention_env_cfg:G1AttentionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1AttentionPPORunnerCfg",
    },
)
