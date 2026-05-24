import gymnasium as gym
from . import agents

gym.register(id='LeggedRLLab-Isaac-Parkour-Depth-Unitree-Go2-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True, kwargs={'env_cfg_entry_point': f'{__name__}.go2_depth_env_cfg:Go2TSDepthEnvCfg', 'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:Go2TSDepthRunnerCfg'})
gym.register(id='LeggedRLLab-Isaac-Parkour-Depth-Unitree-Go2-Play-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True, kwargs={'env_cfg_entry_point': f'{__name__}.go2_depth_env_cfg:Go2TSDepthEnvCfg_PLAY', 'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:Go2TSDepthRunnerCfg'})
gym.register(id='LeggedRLLab-Isaac-Parkour-Depth-Unitree-Go2-Distill-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True, kwargs={'env_cfg_entry_point': f'{__name__}.go2_depth_env_cfg:Go2TSDepthEnvCfg', 'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_distill_cfg:Go2TSDepthDistillRunnerCfg'})
