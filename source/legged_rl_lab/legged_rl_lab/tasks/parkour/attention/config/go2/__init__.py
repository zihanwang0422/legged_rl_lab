import gymnasium as gym

from . import agents


gym.register(
    id="LeggedRLLab-Isaac-Parkour-Attention-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_attention_env_cfg:Go2AttentionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2AttentionPPORunnerCfg",
    },
)

gym.register(
    id="LeggedRLLab-Isaac-Parkour-Attention-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_attention_env_cfg:Go2AttentionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2AttentionPPORunnerCfg",
    },
)
