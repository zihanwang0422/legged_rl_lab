import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils.utils import resolve_nn_activation as get_activation
from .depth_history_encoder import DepthHistoryEncoder
from rsl_rl.utils import unpad_trajectories
'\nActor-Critic for Teacher-Student architecture, with Depth Image Observation.\n'

class ActorCriticTSDepth(nn.Module):

    def __init__(self, num_actor_obs, num_actions, num_privilege_encoder_input, num_latent_dims, num_critic_obs, depth_image_resolution, actor_hidden_dims=[256, 256, 256], critic_hidden_dims=[256, 256, 256], privilege_encoder_hidden_dims=[256, 128], cnn_input_channel=1, cnn_channel_dims=[1, 1, 1], cnn_strides=[1, 1, 1], cnn_fc_layer_dims=[128, 64], combination_mlp_dims=[128, 32], cnn_kernel_sizes=[2, 2, 2], rnn_type='LSTM', rnn_hidden_size=128, rnn_num_layers=1, activation='elu', init_noise_std=1.0, clip_actions=100.0, **kwargs):
        if kwargs:
            print('ActorCritic.__init__ got unexpected arguments, which will be ignored: ' + str([key for key in kwargs.keys()]))
        super().__init__()
        activation_fn = get_activation(activation)
        mlp_input_dim_a = num_actor_obs + num_latent_dims
        mlp_input_dim_c = num_critic_obs
        self.depth_history_encoder = DepthHistoryEncoder(depth_image_resolution, num_actor_obs, num_latent_dims, cnn_input_channel, cnn_channel_dims, cnn_strides, cnn_fc_layer_dims, cnn_kernel_sizes, combination_mlp_dims, rnn_type, rnn_hidden_size, rnn_num_layers, activation_fn)
        privilege_encoder_layers = []
        privilege_encoder_layers.append(nn.Linear(num_privilege_encoder_input, privilege_encoder_hidden_dims[0]))
        privilege_encoder_layers.append(activation_fn)
        for l in range(len(privilege_encoder_hidden_dims)):
            if l == len(privilege_encoder_hidden_dims) - 1:
                privilege_encoder_layers.append(nn.Linear(privilege_encoder_hidden_dims[l], num_latent_dims))
            else:
                privilege_encoder_layers.append(nn.Linear(privilege_encoder_hidden_dims[l], privilege_encoder_hidden_dims[l + 1]))
                privilege_encoder_layers.append(activation_fn)
        self.privilege_encoder = nn.Sequential(*privilege_encoder_layers)
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        actor_layers.append(nn.Hardtanh(-clip_actions, clip_actions))
        self.actor = nn.Sequential(*actor_layers)
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)
        print(f'Depth History Encoder: {self.depth_history_encoder}')
        print(f'Privilege Encoder MLP: {self.privilege_encoder}')
        print(f'Actor MLP: {self.actor}')
        print(f'Critic MLP: {self.critic}')
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for (idx, module) in enumerate((mod for mod in sequential if isinstance(mod, nn.Linear)))]

    def reset(self, dones=None):
        self.depth_history_encoder.reset_hidden_states(dones)

    def detach_hidden_states(self):
        self.depth_history_encoder.detach_hidden_states()

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, depth_image_features, privilege_observations, act_type, hidden_states=None, masks=None):
        if act_type == 'teacher':
            latent = self.privilege_encoder(privilege_observations)
        elif act_type == 'student':
            latent = self.depth_history_encoder(observations, depth_image_features, hidden_states, masks)
            latent = latent.squeeze(0)
            if masks is not None:
                observations = unpad_trajectories(observations, masks)
        else:
            raise ValueError(f'Invalid act_type: {act_type}')
        mean = self.actor(torch.cat((observations, latent), dim=-1))
        std = torch.max(self.std, torch.tensor(1e-06).to(self.std.device))
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, observations, depth_image_features, privilege_observations, act_type, hidden_states=None, masks=None):
        self.update_distribution(observations, depth_image_features, privilege_observations, act_type, hidden_states, masks)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, depth_image_features, **kwargs):
        return self.act_student(observations, depth_image_features, **kwargs)

    def act_student(self, observations, depth_image_features, **kwargs):
        latent = self.depth_history_encoder(observations, depth_image_features)
        actions_mean = self.actor(torch.cat((observations, latent.squeeze(0)), dim=-1))
        return actions_mean

    def act_teacher(self, observations, privilege_observations, **kwargs):
        latent = self.privilege_encoder(privilege_observations)
        actions_mean = self.actor(torch.cat((observations, latent), dim=-1))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def get_hidden_states(self):
        return self.depth_history_encoder.hidden_states