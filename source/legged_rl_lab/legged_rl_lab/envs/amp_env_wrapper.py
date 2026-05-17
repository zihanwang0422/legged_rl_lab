# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for the AMP RSL-RL wrapper.

The implementation now lives in ``amp_env.py`` so the AMP env and its
matching RSL-RL adapter are defined together.
"""

from .amp_env import AmpRslRlVecEnvWrapper
