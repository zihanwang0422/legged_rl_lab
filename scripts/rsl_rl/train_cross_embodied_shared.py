# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Train a shared policy across G1 and Go2 in one Isaac Lab scene.

The mixed scene spawns BOTH Unitree G1 (29 DOF) and Unitree Go2 (12 DOF) in
every environment instance.  The first half of envs run G1; the second half
run Go2.  A single RSL-RL PPO network learns to control both embodiments via a
2-dim robot-type one-hot appended to the observation.

Observation layout (actor, 98 dims, history=1)
  [robot_id(2) | ang_vel(3) | proj_grav(3) | cmd(3)
   | joint_pos(29) | joint_vel(29) | last_action(29)]
  Go2 envs: joint dims 12-28 are zero-padded.

Action layout (29 dims)
  G1 envs: all 29 dims applied to G1.
  Go2 envs: first 12 dims applied to Go2; dims 12-28 are discarded.

Usage
-----
    # Basic – 512 mixed envs (256 G1 + 256 Go2) on CUDA:0
    python scripts/rsl_rl/train_cross_embodied_shared.py --headless

    # Custom size and device
    python scripts/rsl_rl/train_cross_embodied_shared.py \\
        --num_envs 1024 \\
        --device cuda:0 \\
        --max_iterations 20000 \\
        --headless

    # Resume from earlier checkpoint
    python scripts/rsl_rl/train_cross_embodied_shared.py \\
        --resume \\
        --checkpoint /path/to/model_10000.pt \\
        --headless

NOTE: Because Isaac Sim supports only a single simulation stage per process,
both robots share the same simulation.  Use ``train_cross_embodied_dual.py``
for parallel dual-GPU training (separate checkpoints merged into a policy bank).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train shared G1+Go2 policy in a mixed scene (single process)."
)
parser.add_argument("--num_envs", type=int, default=512, help="Total envs (split equally between G1 and Go2).")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max training iterations.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume training from the latest checkpoint in the experiment directory.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to resume from.")
parser.add_argument("--run_name", type=str, default=None, help="Optional tag appended to the run directory.")
parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval (iterations).")
parser.add_argument(
    "--logger", type=str, default="wandb", choices={"wandb", "tensorboard", "neptune"},
    help="Logger module to use."
)
parser.add_argument(
    "--log_project_name", type=str, default="legged-rl-lab",
    help="Name of the logging project when using wandb or neptune."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Suppress USD stage "Unresolved reference prim path" warnings.
# With thousands of envs, Go2's rotor visuals generate tens of thousands of
# these per-env warnings that flood stdout and block training startup.
# ---------------------------------------------------------------------------
# Suppress "Unresolved reference prim path" USD warnings from Go2 rotor visuals.
# These come from go2_description_base.usd referencing visual prims absent in
# go2_description_physics.usd (rotor links + Head_upper/lower).
# omni.log.set_channel_enabled(channel, False) disables the channel entirely.
try:
    import omni.log as _omni_log
    _omni_log.get_log().set_channel_enabled("omni.usd", False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------

import gymnasium as gym
import logging
import os
import torch

from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import legged_rl_lab  # noqa: F401  – registers all custom gym envs including g1go2_mixed

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

logger = logging.getLogger(__name__)

# Task and agent identifiers (registered in g1go2_mixed/__init__.py)
_TASK_ID = "LeggedRLLab-Isaac-Velocity-Flat-G1Go2-Mixed-v0"
_AGENT_ENTRY = "rsl_rl_cfg_entry_point"


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load and configure env + agent cfgs
    # ------------------------------------------------------------------
    env_cfg = load_cfg_from_registry(_TASK_ID, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(_TASK_ID, _AGENT_ENTRY)

    # Apply CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name

    # Apply logger settings
    agent_cfg.logger = args_cli.logger
    if args_cli.logger in {"wandb", "neptune"}:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    # ------------------------------------------------------------------
    # 2. Build log directory
    # ------------------------------------------------------------------
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    print(f"[INFO] Logging to: {log_dir}")

    env_cfg.log_dir = log_dir

    # ------------------------------------------------------------------
    # 3. Resolve checkpoint for resuming
    # ------------------------------------------------------------------
    resume_path: str | None = None
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
        print(f"[INFO] Resuming from explicit checkpoint: {resume_path}")
    elif args_cli.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")

    # ------------------------------------------------------------------
    # 4. Create environment
    # ------------------------------------------------------------------
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(_TASK_ID, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording training videos.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ------------------------------------------------------------------
    # 5. Save config snapshot
    # ------------------------------------------------------------------
    os.makedirs(log_dir, exist_ok=True)
    dump_yaml(os.path.join(log_dir, "env_cfg.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "agent_cfg.yaml"), agent_cfg)

    # ------------------------------------------------------------------
    # 6. Create runner and (optionally) restore checkpoint
    # ------------------------------------------------------------------
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if resume_path is not None:
        runner.load(resume_path)
        print(f"[INFO] Loaded checkpoint: {resume_path}")

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print(f"[INFO] Starting shared-network training: G1 + Go2 mixed scene")
    print(f"       Total envs: {env_cfg.scene.num_envs}  "
          f"(≈{env_cfg.scene.num_envs // 2} G1 + {env_cfg.scene.num_envs - env_cfg.scene.num_envs // 2} Go2)")
    print(f"       Actor obs : 98 dims  |  Critic obs : 101 dims  |  Actions : 29 dims")

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
