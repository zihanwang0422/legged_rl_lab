# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL AMP-PPO agent configuration for Unitree G1 humanoid."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Import the base AMP algorithm config shared across robots
from legged_rl_lab.tasks.locomotion.amp.config.quadruped.unitree_go2.agents.rsl_rl_amp_cfg import (
    RslRlAmpPpoAlgorithmCfg,
)


@configclass
class UnitreeG1AMPFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """AMP-PPO runner config for G1 humanoid flat terrain."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "unitree_g1_amp_flat"

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
        entropy_coef=0.008,
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
            "amp_learning_rate": 1e-4,
            "amp_replay_buffer_size": 1000000,
            # Humanoid requires more emphasis on style to achieve natural gait
            "amp_task_reward_lerp": 0.4,
            "amp_disc_gradient_penalty_coef": 5.0,
            "amp_disc_logit_reg_coef": 0.05,
            "amp_disc_weight_decay": 0.0001,
        },
    )
