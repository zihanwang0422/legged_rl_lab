# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Script to record reference motion data from a trained policy for AMP training.

Usage:
    python scripts/amp/record_reference_motion.py \
        --task LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
        --checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_xxx.pt \
        --num_steps 5000 \
        --output source/legged_rl_lab/legged_rl_lab/data/motions/go2/trot.pt

This runs the trained policy and records the AMP observation features
(joint positions, velocities, foot positions, base velocities) at each step.
The recorded data can then be used as reference motions for AMP training.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record reference motion data for AMP.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=5000, help="Number of steps to record.")
parser.add_argument("--output", type=str, required=True, help="Output path for .pt file.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import legged_rl_lab  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from legged_rl_lab.tasks.locomotion.amp.mdp.observations import (
    amp_joint_pos_rel,
    amp_joint_vel,
    amp_base_lin_vel,
    amp_base_ang_vel,
    amp_foot_positions_base,
)
from isaaclab.managers import SceneEntityCfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Record reference motion data."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable noise for clean recordings
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner and load checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # Get the unwrapped environment
    unwrapped_env = env.unwrapped

    # Determine foot body names
    foot_body_names = ".*_foot"
    robot_cfg = SceneEntityCfg("robot")
    foot_cfg = SceneEntityCfg("robot", body_names=foot_body_names)

    # Record data
    print(f"[INFO] Recording {args_cli.num_steps} steps from {args_cli.num_envs} environments...")
    all_amp_obs = []

    obs = env.get_observations()
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # Compute AMP observations
        joint_pos = amp_joint_pos_rel(unwrapped_env, robot_cfg)
        joint_vel = amp_joint_vel(unwrapped_env, robot_cfg)
        base_lin_vel = amp_base_lin_vel(unwrapped_env, robot_cfg)
        base_ang_vel = amp_base_ang_vel(unwrapped_env, robot_cfg)
        foot_pos = amp_foot_positions_base(unwrapped_env, foot_cfg)

        amp_obs = torch.cat([joint_pos, joint_vel, base_lin_vel, base_ang_vel, foot_pos], dim=-1)
        all_amp_obs.append(amp_obs.cpu())

        if (step + 1) % 500 == 0:
            print(f"  Recorded {step + 1}/{args_cli.num_steps} steps")

    # Concatenate and save
    all_amp_obs = torch.cat(all_amp_obs, dim=0)
    print(f"[INFO] Total recorded frames: {all_amp_obs.shape[0]}, obs_dim: {all_amp_obs.shape[1]}")

    # Create output directory
    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    torch.save({"amp_obs": all_amp_obs}, args_cli.output)
    print(f"[INFO] Saved reference motion data to: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
