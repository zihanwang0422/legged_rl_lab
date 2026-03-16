with open('scripts/amp/play.py', 'r') as f:
    content = f.read()

old_text = "runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)"

new_text = """agent_dict = agent_cfg.to_dict()
    
    # Translate standard rsl_rl policy to local AMP runner dictionary format
    if "policy" in agent_dict:
        pol_cfg = agent_dict.pop("policy")
        agent_dict["actor"] = {
            "class_name": "MLPModel",
            "units": pol_cfg.get("actor_hidden_dims", [512, 256, 128]),
            "activation": "elu"
        }
        agent_dict["critic"] = {
            "class_name": "MLPModel",
            "units": pol_cfg.get("critic_hidden_dims", [512, 256, 128]),
            "activation": "elu"
        }
        agent_dict["distribution_cfg"] = {
            "name": "GaussianDistribution",
            "init_noise_std": pol_cfg.get("init_noise_std", 1.0)
        }
        agent_dict["critic_obs_normalization"] = pol_cfg.get("critic_obs_normalization", False)
        agent_dict["actor_obs_normalization"] = pol_cfg.get("actor_obs_normalization", False)

    runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)"""

content = content.replace(old_text, new_text)

with open('scripts/amp/play.py', 'w') as f:
    f.write(content)
