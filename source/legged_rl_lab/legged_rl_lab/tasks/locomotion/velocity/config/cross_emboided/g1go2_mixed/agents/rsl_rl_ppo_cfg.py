# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CrossEmbodiedG1Go2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner for the cross-embodied G1 + Go2 flat env.

    Network input sizes are inferred at runtime by the runner from the
    TensorDict returned by the env.  No need to hard-code obs dims here.

    Actor input  : 98 dims  (2 robot_id + 96 shared obs, see g1go2_flat_env_cfg)
    Critic input : 101 dims (2 robot_id + 99 privileged obs)
    Action output: 29 dims  (max DOF count; Go2 only uses first 12)
    """

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "cross_embodied_g1go2_flat"
    empirical_normalization = False
    logger = "wandb"
    wandb_project = "legged-rl-lab"

    policy = RslRlPpoActorCriticCfg(
        noise_std_type="scalar",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
    )
