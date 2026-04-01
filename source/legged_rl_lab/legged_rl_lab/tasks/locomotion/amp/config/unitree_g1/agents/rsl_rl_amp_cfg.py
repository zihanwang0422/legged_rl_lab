# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""RSL-RL AMP-PPO agent configuration for Unitree G1 humanoid."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Import the base AMP algorithm config shared across robots
from legged_rl_lab.tasks.locomotion.amp.config.unitree_go2.agents.rsl_rl_amp_cfg import (
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
        entropy_coef=0.002,  # 0.008→0.002: 减弱熵正则，防止action std无界增长
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg={
            # 缩小判别器容量：[1024, 512] → [512, 256]
            # 过大的判别器配合归一化泄漏问题会轻松达到完美区分（disc_acc≈1.0）
            # 缩小容量使其更难记忆简单特征，迫使它关注真正的步态差异
            "amp_discriminator_hidden_dims": [512, 256],
            "amp_discriminator_activation": "relu",
            # 减慢判别器学习，给 policy 更多赶上的机会
            "amp_learning_rate": 1e-5,
            "amp_replay_buffer_size": 1000000,
            # 50% 任务 / 50% 风格：平衡任务与风格梯度
            # 问题背景：disc_acc≈1.0时style_reward是常数（softplus(-0.75)≈1.16/step），
            #   所有 transition 获得相同 style reward → advantage from style ≡ 0。
            #   lerp=0.25时有效梯度只有25%task，太弱；改为0.5使task梯度翻倍，
            #   先让策略学会走路（policy分布向expert靠近），disc_acc才能开始下降。
            "amp_task_reward_lerp": 0.5,
            # gradient penalty 保持标准值
            "amp_disc_gradient_penalty_coef": 5.0,
            # logit_reg 取中间值 0.2：
            # - 0.3 时 logit≈±0.5，softplus 梯度 sigmoid(-0.5)=0.38 ✓ 但梯度稍弱
            # - 0.1 时 logit≈±1.1，softplus 梯度 sigmoid(-1.1)=0.25 ✗ 反而更弱
            # - 0.2 时 logit≈±0.75，softplus 梯度 sigmoid(-0.75)=0.32，平衡点
            "amp_disc_logit_reg_coef": 0.2,
            "amp_disc_weight_decay": 0.0005,
            # reward_scale 降至2.0：scale=3.0时常量style项(≈3×0.387=1.16/step)过大，
            #   淹没task差分信号，value网络偏向拟合style常量而非task结构。
            "amp_reward_scale": 2.0,
        },
    )
