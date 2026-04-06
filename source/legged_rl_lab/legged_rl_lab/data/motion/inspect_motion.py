#!/usr/bin/env python3
# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Motion data inspector / viewer for AMP reference motion files.

Supports .npz (AMASS), .csv (LAFAN1), .npy, .pt, or a directory.

Usage
-----
    python inspect_motion.py <path> [--robot g1] [--no-convert] [--frames N]

Examples
--------
    ·
    python inspect_motion.py LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv
    python inspect_motion.py AMASS_Retargeted_for_G1/g1/CMU --robot g1
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np

# Allow running directly without package install
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "..", "..")
sys.path.insert(0, _SRC)

# ============================================================================
# Joint reorder maps (MuJoCo/AMASS → IsaacLab BFS)
# ============================================================================
_JOINT_REORDER_MAPS = {
    "g1": np.array([
        0, 3, 6, 9, 13, 17,  # left leg
        1, 4, 7, 10, 14, 18,  # right leg
        2, 5, 8,              # waist
        11, 15, 19, 21, 23, 25, 27,  # left arm
        12, 16, 20, 22, 24, 26, 28,  # right arm
    ], dtype=np.int32),
    "go2": None,  # No reordering needed (or TBD)
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP = "─" * 72


def _fmt_arr(arr: np.ndarray, max_cols: int = 8) -> str:
    flat = arr.flatten()
    if flat.size <= max_cols:
        return "[" + "  ".join(f"{v:+.4f}" for v in flat) + "]"
    shown = "  ".join(f"{v:+.4f}" for v in flat[:max_cols])
    return f"[{shown}  … ({flat.size} values)]"


def _stats(arr: np.ndarray) -> str:
    return (
        f"min={arr.min():+.4f}  max={arr.max():+.4f}  "
        f"mean={arr.mean():+.4f}  std={arr.std():.4f}"
    )


def _print_section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific inspectors
# ─────────────────────────────────────────────────────────────────────────────

def inspect_npz(path: str, robot: str = "g1") -> None:
    """Inspect an AMASS-retarg eted .npz file."""
    _print_section(f"FORMAT: AMASS NPZ  |  {os.path.basename(path)}")
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    print(f"  Keys : {keys}")

    fps = float(d["fps"].item()) if "fps" in d else 30.0
    print(f"  FPS  : {fps}")

    for k in keys:
        if k in ("fps",):
            continue
        v = d[k]
        if v.dtype.kind in ("U", "S"):          # string arrays
            print(f"\n  [{k}]  dtype={v.dtype}  shape={v.shape}")
            for i, name in enumerate(v):
                print(f"    [{i:2d}] {name}")
        else:
            arr = v.astype(np.float32)
            print(f"\n  [{k}]  dtype={v.dtype}  shape={v.shape}")
            print(f"    {_stats(arr)}")
            print(f"    frame[0] : {_fmt_arr(arr[0] if arr.ndim >= 1 else arr)}")

    # Summarise AMP-relevant dimensions
    if "dof_positions" in d and "body_names" in d:
        J = d["dof_positions"].shape[1]
        B = d["body_positions"].shape[1] if "body_positions" in d else 0
        ankle_bodies = [n for n in d["body_names"] if "ankle_roll" in n]
        ankle_indices = [list(d["body_names"]).index(n) for n in ankle_bodies]
        
        # Joint reordering info
        reorder_map = _JOINT_REORDER_MAPS.get(robot, None)
        if reorder_map is not None:
            print(f"\n  ⚠️  JOINT REORDERING APPLIED: MuJoCo/AMASS → IsaacLab BFS")
            print(f"      Reorder map: {reorder_map.tolist()}")
            if "dof_names" in d:
                print(f"      Example mappings:")
                for i in [0, 6, 12]:  # left_hip_pitch, right_hip_pitch, waist_yaw
                    if i < len(d["dof_names"]):
                        print(f"        AMASS[{i:2d}] {str(d['dof_names'][i]):30s} → IsaacLab[{reorder_map[i]:2d}]")
        else:
            print(f"\n  ℹ️  No joint reordering (order matches IsaacLab or not applicable)")
        _print_section("AMP Feature Layout (G1 defaults)")
        print(f"  joint_pos_rel   : {J:3d} dims  (dof_positions – default)")
        print(f"  joint_vel       : {J:3d} dims  (dof_velocities)")
        print(f"  base_lin_vel    :   3 dims  (pelvis lin vel, base frame)")
        print(f"  base_ang_vel    :   3 dims  (pelvis ang vel, base frame)")
        print(f"  foot_bodies     : {ankle_bodies}  →  indices {ankle_indices}")
        print(f"  foot_pos_base   : {len(ankle_indices)*3:3d} dims  ({len(ankle_indices)}×3)")
        total = J + J + 3 + 3 + len(ankle_indices) * 3
        print(f"  ─────────────────────────────")
        print(f"  TOTAL per frame : {total:3d} dims")
        print(f"  With history=2  : {total*2:3d} dims  (discriminator input)")

    print()


def inspect_csv(path: str, robot: str = "g1") -> None:
    """Inspect a LAFAN1-style CSV file."""
    _print_section(f"FORMAT: LAFAN1 CSV  |  {os.path.basename(path)}")
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    N, C = data.shape
    print(f"  Shape : ({N} frames, {C} columns)")
    print(f"  FPS   : 30 (LAFAN1 standard)")

    # Interpret columns for G1 (7 root + 29 joints = 36)
    if C == 36:
        print(f"\n  Column layout (G1):")
        print(f"    [0:3]   root_pos      xyz (world)  {_fmt_arr(data[0, :3])}")
        print(f"    [3:7]   root_quat     qx,qy,qz,qw {_fmt_arr(data[0, 3:7])}")
        print(f"    [7:36]  joint_pos     29 DoFs      {_fmt_arr(data[0, 7:16])}")
        print(f"\n  Statistics per column group:")
        print(f"    root_pos   : {_stats(data[:, :3])}")
        print(f"    root_quat  : {_stats(data[:, 3:7])}")
        print(f"    joint_pos  : {_stats(data[:, 7:])}")
        _print_section("AMP Feature Layout after conversion")
        print(f"  joint_pos_rel  :  29 dims  (CSV joints – default posture)")
        print(f"  joint_vel      :  29 dims  (finite difference × 30 FPS)")
        print(f"  base_lin_vel   :   3 dims  (root pos FD → base frame)")
        print(f"  base_ang_vel   :   3 dims  (quat FD)")
        print(f"  foot_pos_base  :   6 dims  ⚠️  set to ZERO (no FK from CSV)")
        print(f"  ─────────────────────────────")
        print(f"  TOTAL per frame:  70 dims")
        print(f"  ⚠️  Use AMASS .npz for accurate foot position features.")
    else:
        print(f"  (Unknown robot — {C} columns)")
        print(f"  root  (assumed) : [0:7]   {_fmt_arr(data[0, :7])}")
        print(f"  joints(assumed) : [7:{C}]  {_fmt_arr(data[0, 7:15])}")

    print(f"\n  First frame: {_fmt_arr(data[0])}")
    print()


def inspect_npy(path: str, robot: str = "g1") -> None:
    """Inspect a pre-processed .npy file (amp_obs format)."""
    _print_section(f"FORMAT: NPY  |  {os.path.basename(path)}")
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item()
    if isinstance(raw, dict):
        print(f"  Keys: {list(raw.keys())}")
        if "amp_obs" in raw:
            arr = np.asarray(raw["amp_obs"], dtype=np.float32)
            print(f"\n  amp_obs  shape={arr.shape}")
            print(f"    {_stats(arr)}")
            print(f"    frame[0]: {_fmt_arr(arr[0])}")
    else:
        arr = np.asarray(raw, dtype=np.float32)
        print(f"  shape={arr.shape},  dtype={arr.dtype}")
        print(f"  {_stats(arr)}")
        print(f"  frame[0]: {_fmt_arr(arr[0] if arr.ndim > 1 else arr)}")
    print()


def inspect_pt(path: str, robot: str = "g1") -> None:
    """Inspect a pre-processed .pt/.pth file (amp_obs format)."""
    import torch
    _print_section(f"FORMAT: PT  |  {os.path.basename(path)}")
    raw = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(raw, dict):
        print(f"  Keys: {list(raw.keys())}")
        if "amp_obs" in raw:
            t = raw["amp_obs"]
            print(f"\n  amp_obs  shape={tuple(t.shape)}")
            t_np = t.numpy().astype(np.float32)
            print(f"    {_stats(t_np)}")
            print(f"    frame[0]: {_fmt_arr(t_np[0])}")
    else:
        t_np = raw.numpy().astype(np.float32)
        print(f"  shape={t_np.shape}")
        print(f"  {_stats(t_np)}")
        print(f"  frame[0]: {_fmt_arr(t_np[0] if t_np.ndim > 1 else t_np)}")
    print()


def inspect_directory(path: str, robot: str, frames: int) -> None:
    """Inspect a directory of motion files."""
    _print_section(f"DIRECTORY: {path}")
    supported = (".npz", ".csv", ".npy", ".pt", ".pth")
    all_files = []
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(supported):
                all_files.append(os.path.join(root, f))
    print(f"  Found {len(all_files)} motion files")
    if not all_files:
        print("  (no supported files)")
        return
    print(f"\n  File list (first 20):")
    for fp in all_files[:20]:
        print(f"    {fp[len(path)+1:]}")
    if len(all_files) > 20:
        print(f"    ... and {len(all_files)-20} more")

    # Inspect first file in detail
    first = all_files[0]
    ext = os.path.splitext(first)[-1].lower()
    print(f"\n  Detailed inspection of first file:")
    inspect_file(first, robot, frames)

    # Convert and report total AMP obs shape
    if frames > 0:
        print(f"\n  Loading up to {min(frames, 3)} files to count total frames:")
        try:
            sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", ".."))
            sys.path.insert(
                0,
                os.path.join(
                    _HERE,
                    "../../tasks/locomotion/amp",
                ),
            )
            from motion_loader import MotionLoader
            loader = MotionLoader(device="cpu", robot=robot)
            total_frames = 0
            for fp in all_files[:3]:
                t = loader.load(fp)
                total_frames += t.shape[0]
            print(f"    AMP obs shape (first 3 files): N={total_frames} × D={t.shape[1]}")
        except Exception as e:
            print(f"    Could not run conversion: {e}")


def inspect_file(path: str, robot: str, frames: int) -> None:
    """Dispatch to the right inspector based on extension."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npz":
        inspect_npz(path, robot)
    elif ext == ".csv":
        inspect_csv(path, robot)
    elif ext == ".npy":
        inspect_npy(path, robot)
    elif ext in (".pt", ".pth"):
        inspect_pt(path, robot)
    else:
        print(f"  Unsupported extension: {ext}")
        return

    # Optionally show the converted AMP obs
    if frames > 0 and ext in (".npz", ".csv"):
        print(f"\n  Converting to AMP features (first {frames} frames)...")
        try:
            _motions_dir = os.path.join(
                _HERE, "..", "..", "..", "source", "legged_rl_lab",
                "legged_rl_lab", "tasks", "locomotion", "amp"
            )
            # Try the package import path first
            sys.path.insert(0, os.path.join(_HERE, "../../tasks/locomotion/amp"))
            from motion_loader import MotionLoader
            loader = MotionLoader(device="cpu", robot=robot)
            tensor = loader.load(path)
            print(f"  → AMP obs tensor: shape={tuple(tensor.shape)}")
            arr = tensor.numpy()
            print(f"     {_stats(arr)}")
            if arr.shape[0] >= 1:
                print(f"     frame[0]: {_fmt_arr(arr[0])}")
            if arr.shape[0] >= 2:
                print(f"     frame[1]: {_fmt_arr(arr[1])}")
        except Exception as e:
            print(f"  (Conversion failed: {e})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Motion file or directory to inspect.")
    parser.add_argument(
        "--robot", default="g1",
        choices=["g1", "go2", "h1"],
        help="Robot type (used for column/body index interpretation). Default: g1",
    )
    parser.add_argument(
        "--frames", type=int, default=2,
        help="Number of converted AMP obs frames to show (0 = skip conversion). Default: 2",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.exists(path):
        print(f"Error: path does not exist: {path}")
        sys.exit(1)

    print(f"\n{'═'*72}")
    print(f"  Motion Inspector")
    print(f"  path  : {path}")
    print(f"  robot : {args.robot}")
    print(f"{'═'*72}")

    if os.path.isdir(path):
        inspect_directory(path, args.robot, args.frames)
    else:
        inspect_file(path, args.robot, args.frames)


if __name__ == "__main__":
    main()
