#!/usr/bin/env python3
# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""Verify joint ordering alignment between LAFAN1 CSV data and IsaacLab.

This script:
1. Loads a LAFAN1 CSV file via MotionLoader (applies joint reordering)
2. Prints the LAFAN1/AMASS DFS joint order (from CSV)
3. Prints the IsaacLab BFS joint order (expected by the environment)
4. Shows the reorder mapping and verifies correctness
5. Prints basic statistics of the loaded motion data

Usage:
    python scripts/amp/verify_joint_order.py
"""

import sys
import os
import numpy as np

# Add project paths - import motion_loader directly to avoid IsaacLab dependency
scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(scripts_dir, "../.."))
source_dir = os.path.join(project_root, "source/legged_rl_lab")

# Import motion_loader directly (no IsaacLab dependency)
import importlib.util
motion_loader_path = os.path.join(source_dir, "legged_rl_lab/managers/motion_loader.py")
spec = importlib.util.spec_from_file_location("motion_loader", motion_loader_path)
motion_loader_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion_loader_mod)
MotionLoader = motion_loader_mod.MotionLoader
_ROBOT_PROFILES = motion_loader_mod._ROBOT_PROFILES


def main():
    # ---- Joint name definitions ----
    # LAFAN1 CSV / AMASS / MuJoCo DFS order (from LAFAN1 README)
    amass_joint_names = [
        "left_hip_pitch_joint",       # 0
        "left_hip_roll_joint",        # 1
        "left_hip_yaw_joint",         # 2
        "left_knee_joint",            # 3
        "left_ankle_pitch_joint",     # 4
        "left_ankle_roll_joint",      # 5
        "right_hip_pitch_joint",      # 6
        "right_hip_roll_joint",       # 7
        "right_hip_yaw_joint",        # 8
        "right_knee_joint",           # 9
        "right_ankle_pitch_joint",    # 10
        "right_ankle_roll_joint",     # 11
        "waist_yaw_joint",            # 12
        "waist_roll_joint",           # 13
        "waist_pitch_joint",          # 14
        "left_shoulder_pitch_joint",  # 15
        "left_shoulder_roll_joint",   # 16
        "left_shoulder_yaw_joint",    # 17
        "left_elbow_joint",           # 18
        "left_wrist_roll_joint",      # 19
        "left_wrist_pitch_joint",     # 20
        "left_wrist_yaw_joint",       # 21
        "right_shoulder_pitch_joint", # 22
        "right_shoulder_roll_joint",  # 23
        "right_shoulder_yaw_joint",   # 24
        "right_elbow_joint",          # 25
        "right_wrist_roll_joint",     # 26
        "right_wrist_pitch_joint",    # 27
        "right_wrist_yaw_joint",      # 28
    ]

    # IsaacLab BFS order (from UNITREE_G1_29DOF_CFG default_joint_pos)
    bfs_joint_names = [
        "left_hip_pitch_joint",       # 0
        "right_hip_pitch_joint",      # 1
        "waist_yaw_joint",            # 2
        "left_hip_roll_joint",        # 3
        "right_hip_roll_joint",       # 4
        "waist_roll_joint",           # 5
        "left_hip_yaw_joint",         # 6
        "right_hip_yaw_joint",        # 7
        "waist_pitch_joint",          # 8
        "left_knee_joint",            # 9
        "right_knee_joint",           # 10
        "left_shoulder_pitch_joint",  # 11
        "right_shoulder_pitch_joint", # 12
        "left_ankle_pitch_joint",     # 13
        "right_ankle_pitch_joint",    # 14
        "left_shoulder_roll_joint",   # 15
        "right_shoulder_roll_joint",  # 16
        "left_ankle_roll_joint",      # 17
        "right_ankle_roll_joint",     # 18
        "left_shoulder_yaw_joint",    # 19
        "right_shoulder_yaw_joint",   # 20
        "left_elbow_joint",           # 21
        "right_elbow_joint",          # 22
        "left_wrist_roll_joint",      # 23
        "right_wrist_roll_joint",     # 24
        "left_wrist_pitch_joint",     # 25
        "right_wrist_pitch_joint",    # 26
        "left_wrist_yaw_joint",       # 27
        "right_wrist_yaw_joint",      # 28
    ]

    # ---- Print joint orders ----
    print("=" * 80)
    print("JOINT ORDER COMPARISON")
    print("=" * 80)

    print(f"\n{'Index':>5}  {'LAFAN1/AMASS (DFS)':40s}  {'IsaacLab (BFS)':40s}")
    print("-" * 90)
    for i in range(29):
        print(f"  [{i:2d}]  {amass_joint_names[i]:40s}  {bfs_joint_names[i]:40s}")

    # ---- Verify reorder map ----
    profile = _ROBOT_PROFILES["g1"]
    gather_map = profile["joint_reorder_map"]

    print(f"\n{'=' * 80}")
    print("REORDER MAP VERIFICATION (gather_map: BFS[i] = AMASS[map[i]])")
    print("=" * 80)
    print(f"\nGather map: {gather_map.tolist()}\n")

    all_correct = True
    for bfs_idx in range(29):
        amass_idx = gather_map[bfs_idx]
        amass_name = amass_joint_names[amass_idx]
        expected_name = bfs_joint_names[bfs_idx]
        match = amass_name == expected_name
        mark = "OK" if match else "MISMATCH"
        if not match:
            all_correct = False
        print(f"  BFS[{bfs_idx:2d}] {expected_name:40s} <- AMASS[{amass_idx:2d}] {amass_name:40s} [{mark}]")

    print(f"\n  Result: {'ALL CORRECT' if all_correct else 'MISMATCHES FOUND!'}\n")

    # ---- Load LAFAN1 walk data ----
    lafan_dir = os.path.join(
        project_root, "source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk"
    )

    if not os.path.exists(lafan_dir):
        print(f"[SKIP] LAFAN1 walk directory not found: {lafan_dir}")
        return

    print("=" * 80)
    print("LAFAN1 WALK DATA LOADING TEST")
    print("=" * 80)

    loader = MotionLoader(device="cpu", robot="g1")
    data = loader.load(lafan_dir)

    print(f"\n  Total frames loaded: {loader.num_frames}")
    print(f"  Observation dim:     {loader.obs_dim}")
    print(f"  Data shape:          {data.shape}")

    # AMP obs layout: joint_pos_rel(29) | joint_vel(29) | base_lin_vel(3) | base_ang_vel(3) | foot_pos(2*3=6)
    expected_dim = 29 + 29 + 3 + 3 + 2 * 3
    print(f"  Expected obs dim:    {expected_dim}")
    dim_ok = loader.obs_dim == expected_dim
    print(f"  Dimension check:     {'OK' if dim_ok else f'MISMATCH (got {loader.obs_dim})'}")

    # Print feature ranges
    print(f"\n  Feature ranges (first frame):")
    frame = data[0].numpy()
    labels = ["joint_pos_rel", "joint_vel", "base_lin_vel", "base_ang_vel", "foot_pos_b"]
    offsets = [0, 29, 58, 61, 64]
    sizes = [29, 29, 3, 3, 6]
    for label, off, sz in zip(labels, offsets, sizes):
        chunk = frame[off:off + sz]
        print(f"    {label:20s}: min={chunk.min():8.4f}  max={chunk.max():8.4f}  mean={chunk.mean():8.4f}")

    # Test sampling
    print(f"\n  Sampling test (history_length=2):")
    sample = loader.sample(batch_size=4, history_length=2)
    print(f"    Sample shape: {sample.shape}  (expected: (4, {expected_dim * 2}))")

    print(f"\n{'=' * 80}")
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
