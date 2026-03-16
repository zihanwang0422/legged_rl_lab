with open('scripts/amp/play.py', 'r') as f:
    content = f.read()

old_text = '"""Script to play/visualise a trained AMP-PPO policy."""'
new_text = '"""Script to play/visualise a trained AMP-PPO policy."""\nimport uuid\nuuid_value = uuid.uuid4()'

old_text_2 = 'runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)'
new_text_2 = '''
    # Add std_param to the action distribution to prevent Unexpected key(s) in state_dict: "distribution.std_param"
    # the local AMPRunner adds standard deviation parameters differently compared to standard ActorCritic config.
    if 'noise_std_type' in pol_cfg and pol_cfg['noise_std_type'] == 'scalar':
         agent_dict["distribution_cfg"].update({
             "init_noise_std": pol_cfg.get("init_noise_std", 1.0)
         })
         # Ensure local custom MLP models are fully configured
    
    runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
'''

content = content.replace(old_text, new_text)
content = content.replace(old_text_2, new_text_2)

with open('scripts/amp/play.py', 'w') as f:
    f.write(content)
