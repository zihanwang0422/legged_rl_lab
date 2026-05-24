from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from legged_rl_lab.rsl_rl import RslRlTsDepthActorCriticCfg, RslRlTsDepthAlgorithmCfg

@configclass
class G1TSDepthDistillRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = 'TsDepthRunner'
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = 'g1_parkour_depth_distill'
    obs_groups = {'policy': ['policy'], 'privileged': ['privileged'], 'depth': ['depth'], 'critic': ['critic']}
    policy = RslRlTsDepthActorCriticCfg(class_name='ActorCriticTSDepth', init_noise_std=1.0, actor_obs_normalization=False, critic_obs_normalization=False, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[1024, 256, 128], activation='elu', clip_actions=10.0, depth_shape=(1, 30, 40), num_latent_dims=64, num_student_envs=None, privilege_encoder_hidden_dims=[256, 128], cnn_input_channel=1, cnn_channel_dims=[8, 8], cnn_strides=[1, 1], cnn_fc_layer_dims=[128, 64], combination_mlp_dims=[128, 32], cnn_kernel_sizes=[5, 3], rnn_type='gru', rnn_hidden_size=512, rnn_num_layers=1)
    algorithm = RslRlTsDepthAlgorithmCfg(class_name='PPO_TSDepth', distillation=True, teacher_checkpoint_path='', value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.0, num_learning_epochs=5, num_mini_batches=4, learning_rate=0.001, encoder_lr=0.0002, schedule='adaptive', gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)