# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL AMP-PPO agent configuration for Unitree Go2."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config extended with AMP discriminator configuration."""

    class_name: str = "AMPPPO"

    amp_cfg: dict | None = None
    """AMP-specific configuration dict. Fields:
        - amp_discriminator_hidden_dims: list[int] (default [1024, 512])
        - amp_discriminator_activation: str (default "relu")
        - amp_learning_rate: float (default 1e-3)
        - amp_replay_buffer_size: int (default 1000000)
        - amp_task_reward_lerp: float (default 0.5, blend factor for task vs style reward)
        - amp_disc_gradient_penalty_coef: float (default 5.0)
        - amp_disc_logit_reg_coef: float (default 0.05)
        - amp_disc_weight_decay: float (default 0.0001)
    """


@configclass
class UnitreeGo2AMPRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """AMP-PPO runner config for Go2 rough terrain."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2_amp_rough"

    policy = RslRlPpoActorCriticCfg(
        noise_std_type="scalar",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlAmpPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg={
            "amp_discriminator_hidden_dims": [1024, 512],
            "amp_discriminator_activation": "relu",
            "amp_learning_rate": 1e-5,
            "amp_replay_buffer_size": 1000000,
            "amp_task_reward_lerp": 0.5,
            "amp_disc_gradient_penalty_coef": 5.0,
            "amp_disc_logit_reg_coef": 0.05,
            "amp_disc_weight_decay": 0.0001,
        },
    )


@configclass
class UnitreeGo2AMPFlatPPORunnerCfg(UnitreeGo2AMPRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "unitree_go2_amp_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
