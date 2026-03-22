#!/usr/bin/env python3
# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Launch G1 and Go2 training on two GPUs and package outputs into one policy bank.

This script orchestrates two independent training processes:
- G1 task on GPU0 (logical cuda:0 inside that process)
- Go2 task on GPU1 (logical cuda:0 inside that process)

After both jobs finish successfully, it saves a single combined checkpoint file
("policy bank") that contains both trained checkpoints for unified deployment
entry points.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


DEFAULT_G1_TASK = "LeggedRLLab-Isaac-Velocity-Flat-Unitree-G1-v0"
DEFAULT_GO2_TASK = "LeggedRLLab-Isaac-Velocity-Flat-Unitree-Go2-v0"


def _build_train_cmd(
    script_path: Path,
    task: str,
    num_envs: int,
    max_iterations: int | None,
    seed: int | None,
    run_name: str,
    headless: bool,
    extra_args: Sequence[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--task",
        task,
        "--device",
        "cuda:0",
        "--num_envs",
        str(num_envs),
        "--run_name",
        run_name,
    ]
    if max_iterations is not None:
        cmd += ["--max_iterations", str(max_iterations)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if headless:
        cmd.append("--headless")
    cmd.extend(extra_args)
    return cmd


def _latest_checkpoint_from_experiment(log_root: Path, experiment_name: str) -> Path:
    exp_dir = log_root / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    run_dirs = [p for p in exp_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {exp_dir}")

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    ckpts = list(latest_run.glob("model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt found in latest run: {latest_run}")

    def _step_id(path: Path) -> int:
        match = re.match(r"model_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    return max(ckpts, key=_step_id)


def _save_policy_bank(output_path: Path, g1_ckpt_path: Path, go2_ckpt_path: Path, g1_task: str, go2_task: str):
    import torch

    g1_payload = torch.load(g1_ckpt_path, map_location="cpu")
    go2_payload = torch.load(go2_ckpt_path, map_location="cpu")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank = {
        "format": "cross_embodied_policy_bank_v1",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "tasks": {"g1": g1_task, "go2": go2_task},
        "source_checkpoints": {"g1": str(g1_ckpt_path), "go2": str(go2_ckpt_path)},
        "payload": {"g1": g1_payload, "go2": go2_payload},
    }
    torch.save(bank, output_path)


def _terminate_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="Dual-GPU cross-embodied training launcher (G1 + Go2).")
    parser.add_argument("--g1-task", type=str, default=DEFAULT_G1_TASK)
    parser.add_argument("--go2-task", type=str, default=DEFAULT_GO2_TASK)
    parser.add_argument("--g1-gpu", type=int, default=0, help="Physical GPU index for G1 training process.")
    parser.add_argument("--go2-gpu", type=int, default=1, help="Physical GPU index for Go2 training process.")
    parser.add_argument("--g1-num-envs", type=int, default=2048)
    parser.add_argument("--go2-num-envs", type=int, default=2048)
    parser.add_argument("--g1-max-iterations", type=int, default=None)
    parser.add_argument("--go2-max-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Base seed; Go2 uses seed+1 if provided.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run Isaac Sim headless.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/rsl_rl"),
        help="Root directory where rsl_rl logs are stored.",
    )
    parser.add_argument(
        "--g1-experiment-name",
        type=str,
        default="unitree_g1_flat",
        help="Experiment folder name for G1 logs.",
    )
    parser.add_argument(
        "--go2-experiment-name",
        type=str,
        default="unitree_go2_flat",
        help="Experiment folder name for Go2 logs.",
    )
    parser.add_argument(
        "--policy-bank-output",
        type=Path,
        default=None,
        help="Optional output path for combined policy bank file. If not set, auto-generate under logs/rsl_rl/policy_bank/.",
    )

    # Remaining arguments are forwarded to both train.py calls.
    args, passthrough = parser.parse_known_args()

    if args.g1_gpu == args.go2_gpu:
        raise ValueError("g1-gpu and go2-gpu must be different for parallel dual-GPU training.")

    project_root = Path(__file__).resolve().parents[2]
    train_script = project_root / "scripts" / "rsl_rl" / "train.py"

    if not train_script.exists():
        raise FileNotFoundError(f"Cannot find train entry: {train_script}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    g1_run_name = f"cross_embodied_g1_{stamp}"
    go2_run_name = f"cross_embodied_go2_{stamp}"

    g1_cmd = _build_train_cmd(
        script_path=train_script,
        task=args.g1_task,
        num_envs=args.g1_num_envs,
        max_iterations=args.g1_max_iterations,
        seed=args.seed,
        run_name=g1_run_name,
        headless=args.headless,
        extra_args=passthrough,
    )

    go2_seed = None if args.seed is None else args.seed + 1
    go2_cmd = _build_train_cmd(
        script_path=train_script,
        task=args.go2_task,
        num_envs=args.go2_num_envs,
        max_iterations=args.go2_max_iterations,
        seed=go2_seed,
        run_name=go2_run_name,
        headless=args.headless,
        extra_args=passthrough,
    )

    env_g1 = os.environ.copy()
    env_g1["CUDA_VISIBLE_DEVICES"] = str(args.g1_gpu)

    env_go2 = os.environ.copy()
    env_go2["CUDA_VISIBLE_DEVICES"] = str(args.go2_gpu)

    print("=" * 80)
    print("Dual training launch")
    print(f"  G1  -> physical gpu {args.g1_gpu}, cmd: {' '.join(g1_cmd)}")
    print(f"  Go2 -> physical gpu {args.go2_gpu}, cmd: {' '.join(go2_cmd)}")
    print("=" * 80)

    g1_proc = subprocess.Popen(g1_cmd, cwd=project_root, env=env_g1)
    # Small delay avoids simultaneous heavy startup spikes.
    time.sleep(2.0)
    go2_proc = subprocess.Popen(go2_cmd, cwd=project_root, env=env_go2)

    def _handle_signal(signum, _frame):
        print(f"\n[WARN] Caught signal {signum}, terminating child processes...")
        _terminate_process(g1_proc)
        _terminate_process(go2_proc)
        raise SystemExit(1)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while True:
        g1_ret = g1_proc.poll()
        go2_ret = go2_proc.poll()

        if g1_ret is not None and g1_ret != 0:
            _terminate_process(go2_proc)
            raise RuntimeError(f"G1 training failed with code {g1_ret}")

        if go2_ret is not None and go2_ret != 0:
            _terminate_process(g1_proc)
            raise RuntimeError(f"Go2 training failed with code {go2_ret}")

        if g1_ret == 0 and go2_ret == 0:
            break

        time.sleep(5.0)

    print("[INFO] Both training jobs finished successfully.")

    g1_ckpt = _latest_checkpoint_from_experiment(args.log_root, args.g1_experiment_name)
    go2_ckpt = _latest_checkpoint_from_experiment(args.log_root, args.go2_experiment_name)

    if args.policy_bank_output is None:
        output_dir = args.log_root / "policy_bank"
        output_path = output_dir / f"cross_embodied_g1_go2_{stamp}.pt"
    else:
        output_path = args.policy_bank_output

    _save_policy_bank(output_path, g1_ckpt, go2_ckpt, args.g1_task, args.go2_task)

    print("[INFO] Policy bank saved:")
    print(f"       {output_path}")
    print("[INFO] Source checkpoints:")
    print(f"       G1 : {g1_ckpt}")
    print(f"       Go2: {go2_ckpt}")


if __name__ == "__main__":
    main()
