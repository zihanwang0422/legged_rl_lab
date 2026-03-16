with open('scripts/amp/play.py', 'r') as f:
    content = f.read()

old_text = 'env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)'
new_text = '''if hasattr(args_cli, "motion_file") and args_cli.motion_file is not None:
        env_cfg.amp_motion_files = args_cli.motion_file

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)'''

content = content.replace(old_text, new_text)

with open('scripts/amp/play.py', 'w') as f:
    f.write(content)
