import torch
state_dict = torch.load("logs/rsl_rl/unitree_g1_amp_flat/2026-03-16_11-07-35/model_1200.pt")
print("Actor dict:")
print(list(state_dict['actor_state_dict'].keys())[:10])
