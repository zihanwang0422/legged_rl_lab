# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from legged_rl_lab.tasks.locomotion.cross_emboided.mdp.cross_procedural_mdp import CrossEmbodiedEncoderCfg


# ---------------------------------------------------------------------------
# Policy cfg variants with encoder support
# ---------------------------------------------------------------------------


@configclass
class CrossEmbodiedActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic cfg that supports a pluggable observation encoder.

    Set ``class_name = "rsl_rl.modules.ActorCriticWithEncoder"`` and fill in
    ``encoder_cfg`` to activate an encoder.  Leave ``encoder_cfg=None`` (or
    use a plain ``RslRlPpoActorCriticCfg``) to fall back to the standard MLP.
    """

    class_name: str = "rsl_rl.modules.ActorCriticWithEncoder"
    encoder_cfg: CrossEmbodiedEncoderCfg | None = None


@configclass
class ProceduralMixedBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Base PPO runner configuration for mixed biped + quadruped training."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = "procedural_mixed"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        noise_std_type="scalar",
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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
class ProceduralMixedFlatPPORunnerCfg(ProceduralMixedBasePPORunnerCfg):
    """PPO runner configuration for mixed procedural robots on flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "procedural_mixed_flat"
        self.max_iterations = 20000


@configclass
class ProceduralMixedRoughPPORunnerCfg(ProceduralMixedBasePPORunnerCfg):
    """PPO runner configuration for mixed procedural robots on rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "procedural_mixed_rough"
        self.max_iterations = 30000


# ---------------------------------------------------------------------------
# Transformer-encoder variants  (flat / rough)
# ---------------------------------------------------------------------------

_TRANSFORMER_POLICY = CrossEmbodiedActorCriticCfg(
    noise_std_type="scalar",
    init_noise_std=1.0,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    # Smaller post-encoder MLP heads — encoder already extracts rich features.
    actor_hidden_dims=[256, 128],
    critic_hidden_dims=[256, 128],
    activation="elu",
    encoder_cfg=CrossEmbodiedEncoderCfg(
        type="transformer",
        latent_dim=256,
        # Obs layout for mixed-flat env:
        #   state = ang_vel(3)+proj_grav(3)+vel_cmds(3) = 9
        #   n_joints = 26 (max between biped-26 and quadruped-12)
        #   morph = 11, height = 0
        state_dim=9,
        n_joints=26,
        morph_dim=11,
        height_dim=0,
        tf_d_model=64,
        tf_n_heads=4,
        tf_n_layers=3,
        tf_dropout=0.0,
    ),
)


@configclass
class ProceduralMixedFlatTransformerPPORunnerCfg(ProceduralMixedBasePPORunnerCfg):
    """Flat terrain: transformer obs encoder."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "procedural_mixed_flat_transformer"
        self.max_iterations = 20000
        self.policy = _TRANSFORMER_POLICY


@configclass
class ProceduralMixedRoughTransformerPPORunnerCfg(ProceduralMixedBasePPORunnerCfg):
    """Rough terrain: transformer obs encoder (with height_dim=187)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "procedural_mixed_rough_transformer"
        self.max_iterations = 30000
        self.policy = CrossEmbodiedActorCriticCfg(
            noise_std_type="scalar",
            init_noise_std=1.0,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[256, 128],
            critic_hidden_dims=[256, 128],
            activation="elu",
            encoder_cfg=CrossEmbodiedEncoderCfg(
                type="transformer",
                latent_dim=256,
                state_dim=9,
                n_joints=26,
                morph_dim=11,
                height_dim=187,  # 17×11 height scan grid
                tf_d_model=64,
                tf_n_heads=4,
                tf_n_layers=3,
            ),
        )


# ---------------------------------------------------------------------------
# GCN-encoder variants  (flat / rough)
# ---------------------------------------------------------------------------

_GCN_POLICY = CrossEmbodiedActorCriticCfg(
    noise_std_type="scalar",
    init_noise_std=1.0,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=[256, 128],
    critic_hidden_dims=[256, 128],
    activation="elu",
    encoder_cfg=CrossEmbodiedEncoderCfg(
        type="gcn",
        latent_dim=256,
        state_dim=9,
        n_joints=26,
        morph_dim=11,
        height_dim=0,
        gcn_d_model=64,
        gcn_n_layers=2,
    ),
)


@configclass
class ProceduralMixedFlatGCNPPORunnerCfg(ProceduralMixedBasePPORunnerCfg):
    """Flat terrain: GCN obs encoder."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "procedural_mixed_flat_gcn"
        self.max_iterations = 20000
        self.policy = _GCN_POLICY


@configclass
class ProceduralMixedRoughGCNPPORunnerCfg(ProceduralMixedBasePPORunnerCfg):
    """Rough terrain: GCN obs encoder."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "procedural_mixed_rough_gcn"
        self.max_iterations = 30000
        self.policy = CrossEmbodiedActorCriticCfg(
            noise_std_type="scalar",
            init_noise_std=1.0,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[256, 128],
            critic_hidden_dims=[256, 128],
            activation="elu",
            encoder_cfg=CrossEmbodiedEncoderCfg(
                type="gcn",
                latent_dim=256,
                state_dim=9,
                n_joints=26,
                morph_dim=11,
                height_dim=0,  # GCN rough: height not routed through GCN
                gcn_d_model=64,
                gcn_n_layers=2,
            ),
        )
