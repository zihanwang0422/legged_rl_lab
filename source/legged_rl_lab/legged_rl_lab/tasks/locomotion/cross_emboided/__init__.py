# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Cross-embodied locomotion tasks.

Tasks for training a single policy across multiple robot embodiments
(e.g., G1 + Go2, procedural quadrupeds, procedural humanoids).
"""

# Register ActorCriticWithEncoder into rsl_rl.modules so that the RSL-RL
# runner can resolve it via eval("rsl_rl.modules.ActorCriticWithEncoder").
from .mdp.cross_procedural_mdp import register_in_rsl_rl as _register_encoder  # noqa
from .mdp.cross_procedural_mdp import CrossEmbodiedEncoderCfg  # noqa: F401
_register_encoder()

from .config.g1go2_mixed import *  # noqa
from .config.procedural_quadruped import *  # noqa
from .config.procedural_humanoid import *  # noqa
from .config.procedural_mixed import *  # noqa
