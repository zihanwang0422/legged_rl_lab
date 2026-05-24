from __future__ import annotations
import argparse
import copy
import os
import sys
import torch
import torch.nn as nn
RSL_RL_PATH = os.path.join(os.path.dirname(__file__), '../../source/legged_rl_lab/legged_rl_lab/rsl_rl')
RSL_RL_PATH = os.path.abspath(RSL_RL_PATH)
if RSL_RL_PATH not in sys.path:
    sys.path.insert(0, RSL_RL_PATH)
from rsl_rl.modules.depth_history_encoder import DepthHistoryEncoder

class TSDepthPolicyExporter(nn.Module):

    def __init__(self, encoder: DepthHistoryEncoder, actor: nn.Module, depth_C: int, depth_H: int, depth_W: int) -> None:
        super().__init__()
        self.cnn = copy.deepcopy(encoder.cnn)
        self.combination_mlp = copy.deepcopy(encoder.combination_mlp)
        self.rnn = copy.deepcopy(encoder.rnn)
        self.latent_output_mlp = copy.deepcopy(encoder.latent_output_mlp)
        self.actor = copy.deepcopy(actor)
        self.depth_C: int = depth_C
        self.depth_H: int = depth_H
        self.depth_W: int = depth_W

    def forward(self, obs: torch.Tensor, depth_obs: torch.Tensor, gru_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        if depth_obs.dim() == 2:
            depth_img = depth_obs.view(B, self.depth_C, self.depth_H, self.depth_W)
        else:
            depth_img = depth_obs
        depth_features = self.cnn(depth_img)
        combined = torch.cat((obs, depth_features), dim=-1)
        combined = self.combination_mlp(combined)
        rnn_in = combined.unsqueeze(0)
        (rnn_out, gru_hidden_new) = self.rnn(rnn_in, gru_hidden)
        latent = self.latent_output_mlp(rnn_out.squeeze(0))
        action = self.actor(torch.cat((obs, latent), dim=-1))
        return (action, gru_hidden_new)

def _build_actor_from_state_dict(model_sd: dict, clip_actions: float | None) -> nn.Module:
    weight_keys = sorted([k for k in model_sd if k.startswith('actor.') and k.endswith('.weight')], key=lambda k: int(k.split('.')[1]))
    layers: list[nn.Module] = []
    for (i, key) in enumerate(weight_keys):
        idx_str = key.split('.')[1]
        w = model_sd[f'actor.{idx_str}.weight']
        b = model_sd[f'actor.{idx_str}.bias']
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(weight_keys) - 1:
            layers.append(nn.ELU())
    if clip_actions is None:
        clip_actions = 100.0
    layers.append(nn.Hardtanh(-float(clip_actions), float(clip_actions)))
    return nn.Sequential(*layers)

def _infer_depth_history_encoder_kwargs(encoder_sd: dict, depth_image_resolution: tuple[int, int]) -> dict:
    MAX_IDX = 64
    conv_indices: list[int] = []
    linear_indices: list[int] = []
    for idx in range(MAX_IDX):
        key = f'cnn.{idx}.weight'
        if key not in encoder_sd:
            continue
        w = encoder_sd[key]
        if w.dim() == 4:
            conv_indices.append(idx)
        elif w.dim() == 2:
            linear_indices.append(idx)
    if not conv_indices:
        raise ValueError('Could not find any Conv2d layers in cnn.* of the encoder state-dict.')
    if not linear_indices:
        raise ValueError('Could not find any Linear layers in cnn.* of the encoder state-dict.')
    cnn_input_channel = int(encoder_sd[f'cnn.{conv_indices[0]}.weight'].shape[1])
    cnn_channel_dims = [int(encoder_sd[f'cnn.{i}.weight'].shape[0]) for i in conv_indices]
    cnn_kernel_sizes = [int(encoder_sd[f'cnn.{i}.weight'].shape[2]) for i in conv_indices]
    cnn_strides = [1] * len(conv_indices)
    cnn_fc_layer_dims = [int(encoder_sd[f'cnn.{i}.weight'].shape[0]) for i in linear_indices]
    cmb_indices: list[int] = []
    for idx in range(MAX_IDX):
        if f'combination_mlp.{idx}.weight' in encoder_sd:
            cmb_indices.append(idx)
    if not cmb_indices:
        raise ValueError('Could not find any Linear layers in combination_mlp.* of encoder state-dict.')
    combination_mlp_dims = [int(encoder_sd[f'combination_mlp.{i}.weight'].shape[0]) for i in cmb_indices]
    cmb0_in = int(encoder_sd[f'combination_mlp.{cmb_indices[0]}.weight'].shape[1])
    num_obs = cmb0_in - cnn_fc_layer_dims[-1]
    rnn_w_ih_l0 = encoder_sd['rnn.weight_ih_l0']
    if 'rnn.bias_ih_l0' not in encoder_sd:
        raise ValueError('Encoder rnn appears not to be GRU/LSTM (no bias_ih_l0).')
    if rnn_w_ih_l0.shape[0] % 3 == 0 and rnn_w_ih_l0.shape[0] // 3 == int(encoder_sd['rnn.weight_hh_l0'].shape[1]):
        rnn_type = 'gru'
        rnn_hidden_size = int(rnn_w_ih_l0.shape[0] // 3)
    else:
        rnn_type = 'lstm'
        rnn_hidden_size = int(rnn_w_ih_l0.shape[0] // 4)
    rnn_num_layers = 0
    while f'rnn.weight_ih_l{rnn_num_layers}' in encoder_sd:
        rnn_num_layers += 1
    num_latent_dims = int(encoder_sd['latent_output_mlp.0.weight'].shape[0])
    return dict(depth_image_resolution=tuple(depth_image_resolution), num_obs=num_obs, num_latent_dims=num_latent_dims, cnn_input_channel=cnn_input_channel, cnn_channel_dims=cnn_channel_dims, cnn_strides=cnn_strides, cnn_fc_layer_dims=cnn_fc_layer_dims, cnn_kernel_sizes=cnn_kernel_sizes, combination_mlp_dims=combination_mlp_dims, rnn_type=rnn_type, rnn_hidden_size=rnn_hidden_size, rnn_num_layers=rnn_num_layers)

def _infer_depth_resolution(encoder_sd: dict, candidates: list[tuple[int, int]]) -> tuple[int, int]:
    conv_indices = [i for i in range(50) if f'cnn.{i}.weight' in encoder_sd and encoder_sd[f'cnn.{i}.weight'].dim() == 4]
    linear_indices = [i for i in range(50) if f'cnn.{i}.weight' in encoder_sd and encoder_sd[f'cnn.{i}.weight'].dim() == 2]
    if not conv_indices or not linear_indices:
        raise ValueError('Could not identify CNN conv/linear layers.')
    expected_flatten_in = int(encoder_sd[f'cnn.{linear_indices[0]}.weight'].shape[1])
    last_conv_ch = int(encoder_sd[f'cnn.{conv_indices[-1]}.weight'].shape[0])
    kernel_sizes = [int(encoder_sd[f'cnn.{i}.weight'].shape[2]) for i in conv_indices]

    def _spatial_after_cnn(H: int, W: int) -> tuple[int, int]:
        (h, w) = (H, W)
        for (i, k) in enumerate(kernel_sizes):
            h = (h - k) // 1 + 1
            w = (w - k) // 1 + 1
            if i == 0:
                h = (h - 2) // 2 + 1
                w = (w - 2) // 2 + 1
        return (h, w)
    for (H, W) in candidates:
        (h, w) = _spatial_after_cnn(H, W)
        if h * w * last_conv_ch == expected_flatten_in:
            return (H, W)
    raise ValueError(f'None of the candidate depth resolutions {candidates} match the CNN flatten size {expected_flatten_in} (last_conv_ch={last_conv_ch}).')

def export_ts_depth_policy(checkpoint_path: str, output_path: str | None=None, onnx_path: str | None=None, clip_actions: float=10.0, depth_resolution: tuple[int, int] | None=None, candidate_resolutions: list[tuple[int, int]] | None=None) -> tuple[str, str | None]:
    print(f'[INFO] Loading checkpoint: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_sd = ckpt.get('model_state_dict', ckpt)
    encoder_prefix = 'depth_history_encoder.'
    encoder_sd = {k[len(encoder_prefix):]: v for (k, v) in model_sd.items() if k.startswith(encoder_prefix)}
    if not encoder_sd:
        raise ValueError("Checkpoint has no 'depth_history_encoder.*' weights — is this a TS-Depth model?")
    if depth_resolution is None:
        if candidate_resolutions is None:
            candidate_resolutions = [(30, 40), (60, 80), (48, 64), (32, 48), (64, 64), (24, 32)]
        depth_resolution = _infer_depth_resolution(encoder_sd, candidate_resolutions)
        print(f'[INFO] Inferred depth resolution (H, W) = {depth_resolution}')
    else:
        print(f'[INFO] Using provided depth resolution (H, W) = {depth_resolution}')
    enc_kwargs = _infer_depth_history_encoder_kwargs(encoder_sd, depth_resolution)
    print(f'[INFO] Detected DepthHistoryEncoder kwargs:')
    for (k, v) in enc_kwargs.items():
        print(f'  {k} = {v}')
    encoder = DepthHistoryEncoder(activation_fn=nn.ELU(), **enc_kwargs)
    encoder.load_state_dict(encoder_sd)
    print(f'[INFO] Loaded encoder ({len(encoder_sd)} tensors).')
    actor = _build_actor_from_state_dict(model_sd, clip_actions=clip_actions)
    num_actions = int(actor[-2].out_features)
    print(f'[INFO] Built actor: input_dim={actor[0].in_features}, num_actions={num_actions}')
    depth_C = enc_kwargs['cnn_input_channel']
    (depth_H, depth_W) = enc_kwargs['depth_image_resolution']
    num_obs = enc_kwargs['num_obs']
    rnn_num_layers = enc_kwargs['rnn_num_layers']
    rnn_hidden_size = enc_kwargs['rnn_hidden_size']
    exporter = TSDepthPolicyExporter(encoder, actor, depth_C, depth_H, depth_W).eval()
    test_obs = torch.zeros(1, num_obs)
    test_depth_flat = torch.zeros(1, depth_C * depth_H * depth_W)
    test_h = torch.zeros(rnn_num_layers, 1, rnn_hidden_size)
    with torch.inference_mode():
        (ref_action, ref_h) = exporter(test_obs, test_depth_flat, test_h)
    print(f'[INFO] Reference forward — obs: {tuple(test_obs.shape)}, depth: {tuple(test_depth_flat.shape)}, h: {tuple(test_h.shape)} → action: {tuple(ref_action.shape)}, h_out: {tuple(ref_h.shape)}')
    traced = torch.jit.trace(exporter, (test_obs, test_depth_flat, test_h), check_trace=False)
    if output_path is None:
        out_dir = os.path.join(os.path.dirname(checkpoint_path), 'exported')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, 'ts_depth_policy.pt')
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced.save(output_path)
    print(f'[INFO] Exported TorchScript → {output_path}')
    loaded = torch.jit.load(output_path)
    with torch.inference_mode():
        (v_action, v_h) = loaded(test_obs, test_depth_flat, test_h)
    assert torch.allclose(ref_action, v_action, atol=1e-05), 'JIT action mismatch!'
    assert torch.allclose(ref_h, v_h, atol=1e-05), 'JIT GRU hidden mismatch!'
    print('[INFO] JIT verification passed.')
    onnx_out_path: str | None = None
    if onnx_path is not None:
        os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
        torch.onnx.export(exporter, (test_obs, test_depth_flat, test_h), onnx_path, export_params=True, opset_version=18, input_names=['obs', 'depth_obs', 'gru_hidden'], output_names=['action', 'gru_hidden_new'], dynamic_axes={'obs': {0: 'batch'}, 'depth_obs': {0: 'batch'}, 'gru_hidden': {1: 'batch'}, 'action': {0: 'batch'}, 'gru_hidden_new': {1: 'batch'}})
        onnx_out_path = onnx_path
        print(f'[INFO] Exported ONNX → {onnx_path}')
    print('\n[INFO] Deployment parameters:')
    print(f'  num_obs        = {num_obs}')
    print(f'  num_actions    = {num_actions}')
    print(f'  depth_C/H/W    = {depth_C}, {depth_H}, {depth_W}')
    print(f'  gru_num_layers = {rnn_num_layers}')
    print(f'  gru_hidden     = {rnn_hidden_size}')
    print(f'  clip_actions   = {clip_actions}')
    print(f'\nInitial GRU hidden at episode start:')
    print(f'  torch.zeros({rnn_num_layers}, 1, {rnn_hidden_size})')
    return (output_path, onnx_out_path)

def main() -> None:
    parser = argparse.ArgumentParser(description='Export TS-Depth student policy to TorchScript / ONNX.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model_xxxx.pt')
    parser.add_argument('--output', type=str, default=None, help='Output JIT .pt path (default: <checkpoint_dir>/exported/ts_depth_policy.pt)')
    parser.add_argument('--onnx', type=str, nargs='?', const='__default__', default=None, help='Also export ONNX. Pass a path or omit value to write next to the JIT file.')
    parser.add_argument('--clip_actions', type=float, default=10.0, help='Hardtanh clip applied to the actor output (default: 10.0; matches Go2 distill cfg).')
    parser.add_argument('--depth_h', type=int, default=None, help='Depth image height (auto-inferred from CNN flatten if not provided).')
    parser.add_argument('--depth_w', type=int, default=None, help='Depth image width (auto-inferred from CNN flatten if not provided).')
    args = parser.parse_args()
    depth_res: tuple[int, int] | None = None
    if args.depth_h is not None and args.depth_w is not None:
        depth_res = (args.depth_h, args.depth_w)
    onnx_path: str | None = None
    if args.onnx is not None:
        if args.onnx == '__default__':
            ckpt_dir = os.path.dirname(args.checkpoint)
            onnx_path = os.path.join(ckpt_dir, 'exported', 'ts_depth_policy.onnx')
        else:
            onnx_path = args.onnx
    export_ts_depth_policy(args.checkpoint, output_path=args.output, onnx_path=onnx_path, clip_actions=args.clip_actions, depth_resolution=depth_res)
if __name__ == '__main__':
    main()
