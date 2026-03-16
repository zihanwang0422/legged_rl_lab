import urllib.request

# Read the file
with open('scripts/amp/play.py', 'r') as f:
    content = f.read()

import re
old_text = """# reuse cli_args from scripts/rsl_rl
sys.path.insert(0, sys.path[0].replace("/amp", "/rsl_rl"))"""

new_text = """# add path of scripts/rsl_rl so we can reuse cli_args
import os
import pathlib
scripts_dir = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.insert(0, os.path.join(scripts_dir, "rsl_rl"))

# Add custom rsl_rl to sys.path to override the pip-installed version
custom_rsl_rl_dir = os.path.abspath(os.path.join(scripts_dir, "../source/legged_rl_lab/legged_rl_lab/rsl_rl"))
sys.path.insert(0, custom_rsl_rl_dir)"""

content = content.replace(old_text, new_text)

with open('scripts/amp/play.py', 'w') as f:
    f.write(content)
