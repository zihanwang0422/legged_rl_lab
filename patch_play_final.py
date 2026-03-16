with open("scripts/amp/play.py", "r") as f:
    text = f.read()

import re
old_text = re.search(r'agent_dict \= agent_cfg\.to_dict\(\)\n    \n    \# Translate standard rsl_rl policy(.*?)runner \= OnPolicyRunner', text, re.DOTALL)
if old_text:
    new_str = '''agent_dict = agent_cfg.to_dict()
    
    if "policy" in agent_dict:
        policy_cfg = agent_dict.pop("policy")
        agent_dict["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("actor_obs_normalization", False),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "std_type": policy_cfg.get("noise_std_type", "scalar"),
                "init_std": policy_cfg.get("init_noise_std", 1.0),
            },
        }
        agent_dict["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
            "activation": policy_cfg.get("activation", "elu"),
            "obs_normalization": policy_cfg.get("critic_obs_normalization", False),
        }

    runner = OnPolicyRunner'''
    text = text.replace(old_text.group(0), new_str)
    
    with open("scripts/amp/play.py", "w") as f:
        f.write(text)
