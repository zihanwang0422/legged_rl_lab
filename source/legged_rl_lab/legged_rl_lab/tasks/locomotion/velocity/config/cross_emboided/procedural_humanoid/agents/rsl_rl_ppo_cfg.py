# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ProceduralHumanoidBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Base PPO runner configuration for procedural humanoid training."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = "procedural_humanoid"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        noise_std_type="scalar",
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
    )


@configclass
class ProceduralHumanoidFlatPPORunnerCfg(ProceduralHumanoidBasePPORunnerCfg):
    """PPO runner configuration for procedural humanoid on flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "procedural_humanoid_flat"
        self.max_iterations = 20000


@configclass
class ProceduralHumanoidRoughPPORunnerCfg(ProceduralHumanoidBasePPORunnerCfg):
    """PPO runner configuration for procedural humanoid on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "procedural_humanoid_rough"
        self.max_iterations = 30000
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
