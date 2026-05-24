import torch
import torch.nn as nn
from rsl_rl.utils import unpad_trajectories

class DepthHistoryEncoder(nn.Module):

    def __init__(self, depth_image_resolution, num_obs, num_latent_dims, cnn_input_channel, cnn_channel_dims, cnn_strides, cnn_fc_layer_dims, cnn_kernel_sizes, combination_mlp_dims, rnn_type, rnn_hidden_size, rnn_num_layers, activation_fn):
        super().__init__()
        in_channels = cnn_input_channel
        in_height = depth_image_resolution[0]
        in_width = depth_image_resolution[1]
        cnn_layers = []
        for (i, out_channels) in enumerate(cnn_channel_dims):
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=cnn_kernel_sizes[i], stride=cnn_strides[i]))
            if i != 0:
                cnn_layers.append(activation_fn)
            in_channels = out_channels
            in_height = (in_height - cnn_kernel_sizes[i]) // cnn_strides[i] + 1
            in_width = (in_width - cnn_kernel_sizes[i]) // cnn_strides[i] + 1
            if i == 0:
                cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                cnn_layers.append(activation_fn)
                in_height = (in_height - 2) // 2 + 1
                in_width = (in_width - 2) // 2 + 1
        cnn_layers.append(nn.Flatten())
        cnn_out_dim = in_height * in_width * cnn_channel_dims[-1]
        for l in range(len(cnn_fc_layer_dims)):
            if l == 0:
                cnn_layers.append(nn.Linear(cnn_out_dim, cnn_fc_layer_dims[l]))
            else:
                cnn_layers.append(nn.Linear(cnn_fc_layer_dims[l - 1], cnn_fc_layer_dims[l]))
            cnn_layers.append(activation_fn)
        self.cnn = nn.Sequential(*cnn_layers)
        combination_mlp_layers = []
        for l in range(len(combination_mlp_dims)):
            if l == 0:
                combination_mlp_layers.append(nn.Linear(cnn_fc_layer_dims[-1] + num_obs, combination_mlp_dims[l]))
                combination_mlp_layers.append(activation_fn)
            elif l == -1:
                combination_mlp_layers.append(nn.Linear(combination_mlp_dims[l - 1], combination_mlp_dims[l]))
            else:
                combination_mlp_layers.append(nn.Linear(combination_mlp_dims[l - 1], combination_mlp_dims[l]))
                combination_mlp_layers.append(activation_fn)
        self.combination_mlp = nn.Sequential(*combination_mlp_layers)
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=combination_mlp_dims[-1], hidden_size=rnn_hidden_size, num_layers=rnn_num_layers)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=combination_mlp_dims[-1], hidden_size=rnn_hidden_size, num_layers=rnn_num_layers)
        self.latent_output_mlp = nn.Sequential(nn.Linear(rnn_hidden_size, num_latent_dims))
        self.hidden_states = None

    def forward(self, observation, depth_image, hidden_states=None, masks=None):
        batch_mode = masks is not None
        if batch_mode:
            shape0 = depth_image.shape[0]
            shape1 = depth_image.shape[1]
            depth_image = depth_image.flatten(0, 1)
        depth_encoding = self.cnn(depth_image)
        if not batch_mode:
            observation = observation.unsqueeze(0)
            depth_encoding = depth_encoding.unsqueeze(0)
        else:
            depth_encoding = depth_encoding.view(shape0, shape1, -1)
        combined_input = torch.cat((observation, depth_encoding), dim=-1)
        combined_encoding = self.combination_mlp(combined_input)
        if batch_mode:
            if hidden_states is None:
                raise ValueError('Hidden states not passed to memory module during policy update')
            (rnn_out, _) = self.rnn(combined_encoding, hidden_states)
            rnn_out = unpad_trajectories(rnn_out, masks)
        else:
            (rnn_out, self.hidden_states) = self.rnn(combined_encoding, self.hidden_states)
        latent_output = self.latent_output_mlp(rnn_out)
        return latent_output

    def reset_hidden_states(self, dones=None):
        if self.hidden_states is None:
            return
        states = self.hidden_states if isinstance(self.hidden_states, (tuple, list)) else (self.hidden_states,)
        for hidden_state in states:
            hidden_state[..., dones, :] = 0.0

    def detach_hidden_states(self):
        if self.hidden_states is None:
            return
        if isinstance(self.hidden_states, tuple):
            (h, c) = self.hidden_states
            self.hidden_states = (h.detach().clone(), c.detach().clone())
        else:
            h = self.hidden_states
            self.hidden_states = h.detach().clone()