# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Script to play a robot from a cross-embodied policy bank.

Usage:
    # Play G1 policy extracted from bank
    python scripts/rsl_rl/play_cross_embodied_dual.py \\
        --bank logs/rsl_rl/policy_bank/cross_embodied_g1_go2_<ts>.pt \\
        --robot g1 \\
        --num_envs 1 \\
        --headless

    # Play Go2 policy
    python scripts/rsl_rl/play_cross_embodied_dual.py \\
        --bank logs/rsl_rl/policy_bank/cross_embodied_g1_go2_<ts>.pt \\
        --robot go2 \\
        --num_envs 4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import tempfile

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing (must happen before Isaac Sim launch)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Play a robot policy extracted from a cross-embodied policy bank.")
parser.add_argument("--bank", type=str, required=True, help="Path to cross-embodied policy bank (.pt).")
parser.add_argument(
    "--robot",
    type=str,
    required=True,
    choices=["g1", "go2"],
    help="Which robot's policy to play.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during play.")
parser.add_argument("--video_length", type=int, default=300, help="Length of recorded video (steps).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run at real-time speed.")
parser.add_argument(
    "--export",
    action="store_true",
    default=False,
    help="Export the loaded policy to JIT and ONNX after loading.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------

import gymnasium as gym
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401
import legged_rl_lab  # noqa: F401  – registers all custom gym envs

# ---------------------------------------------------------------------------
# Robot → task ID mapping (task IDs registered in g1_exper / go2_exper)
# ---------------------------------------------------------------------------

ROBOT_TASK_MAP: dict[str, str] = {
    "g1": "LeggedRLLab-Isaac-Velocity-Flat-Unitree-G1-v0",
    "go2": "LeggedRLLab-Isaac-Velocity-Flat-Unitree-Go2-v0",
}


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load bank and extract checkpoint
    # ------------------------------------------------------------------
    bank_path = os.path.abspath(args_cli.bank)
    if not os.path.isfile(bank_path):
        raise FileNotFoundError(f"Policy bank not found: {bank_path}")

    bank = torch.load(bank_path, map_location="cpu", weights_only=False)

    fmt = bank.get("format", "")
    if fmt != "cross_embodied_policy_bank_v1":
        raise ValueError(
            f"Unsupported bank format '{fmt}'. Expected 'cross_embodied_policy_bank_v1'."
        )

    payload = bank.get("payload", {})
    if args_cli.robot not in payload:
        available = list(payload.keys())
        raise KeyError(
            f"Bank does not contain a payload for robot '{args_cli.robot}'. "
            f"Available robots: {available}"
        )

    # Prefer task ID stored in the bank; fall back to defaults above.
    task_id = bank.get("tasks", {}).get(args_cli.robot, ROBOT_TASK_MAP[args_cli.robot])
    print(f"[INFO] Playing robot='{args_cli.robot}' | task='{task_id}'")
    print(f"[INFO] Bank: {bank_path}")

    # Write checkpoint to a temporary file so the runner can load it normally.
    ckpt_dict = payload[args_cli.robot]
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(ckpt_dict, tmp.name)
    tmp.close()
    ckpt_path = tmp.name

    try:
        # ------------------------------------------------------------------
        # 2. Build environment
        # ------------------------------------------------------------------
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")

        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device:
            env_cfg.sim.device = args_cli.device

        # Flat floor play – disable terrain curriculum if present.
        if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None

        render_mode = "rgb_array" if args_cli.video else None
        env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)

        if args_cli.video:
            video_dir = os.path.join(os.path.dirname(bank_path), "videos", "play", args_cli.robot)
            video_kwargs = {
                "video_folder": video_dir,
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print(f"[INFO] Recording video → {video_dir}")
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # ------------------------------------------------------------------
        # 3. Load runner and restore weights from bank
        # ------------------------------------------------------------------
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt_path)
        print(f"[INFO] Loaded checkpoint for robot='{args_cli.robot}' from bank.")

        # Grab policy and (optionally) normalizer for export.
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = runner.alg.policy
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)

        if args_cli.export:
            export_dir = os.path.join(os.path.dirname(bank_path), "exported", args_cli.robot)
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
            print(f"[INFO] Exported policy → {export_dir}")

        # ------------------------------------------------------------------
        # 4. Inference loop
        # ------------------------------------------------------------------
        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        timestep = 0

        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            if args_cli.video:
                timestep += 1
                if timestep >= args_cli.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        env.close()

    finally:
        # Always remove the temp checkpoint file.
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)


if __name__ == "__main__":
    main()
    simulation_app.close()
