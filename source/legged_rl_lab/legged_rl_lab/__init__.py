"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

import os
#/home/wzh/amp/legged_rl_lab/source/legged_rl_lab/legged_rl_lab
LEGGED_RL_LAB_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
