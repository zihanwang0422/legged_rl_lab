import gymnasium as gym
from . import agents

gym.register(id='LeggedRLLab-Isaac-Parkour-Depth-Unitree-G1-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True, kwargs={'env_cfg_entry_point': f'{__name__}.g1_depth_env_cfg:G1TSDepthEnvCfg', 'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:G1TSDepthRunnerCfg'})
gym.register(id='LeggedRLLab-Isaac-Parkour-Depth-Unitree-G1-Play-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True, kwargs={'env_cfg_entry_point': f'{__name__}.g1_depth_env_cfg:G1TSDepthEnvCfg_PLAY', 'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_cfg:G1TSDepthRunnerCfg'})
gym.register(id='LeggedRLLab-Isaac-Parkour-Depth-Unitree-G1-Distill-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True, kwargs={'env_cfg_entry_point': f'{__name__}.g1_depth_env_cfg:G1TSDepthEnvCfg', 'rsl_rl_cfg_entry_point': f'{agents.__name__}.rsl_rl_ppo_distill_cfg:G1TSDepthDistillRunnerCfg'})
