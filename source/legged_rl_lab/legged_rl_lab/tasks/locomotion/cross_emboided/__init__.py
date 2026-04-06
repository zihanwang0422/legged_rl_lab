# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Cross-embodied locomotion tasks.

Tasks for training a single policy across multiple robot embodiments
(e.g., G1 + Go2, procedural quadrupeds, procedural humanoids).
"""

from .config.g1go2_mixed import *  # noqa
from .config.procedural_quadruped import *  # noqa
from .config.procedural_humanoid import *  # noqa
