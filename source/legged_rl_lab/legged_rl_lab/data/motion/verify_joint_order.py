#!/usr/bin/env python3
"""Verify joint order mapping between MuJoCo (AMASS) and IsaacLab (BFS) for G1."""

# MuJoCo/AMASS order (from inspect_motion.py output)
MUJOCO_ORDER = [
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

# IsaacLab BFS order (from user comment)
ISAACLAB_ORDER = [
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

# Build mapping: mujoco_to_isaaclab[i] = index in IsaacLab order
mujoco_to_isaaclab = []
for mj_joint in MUJOCO_ORDER:
    il_idx = ISAACLAB_ORDER.index(mj_joint)
    mujoco_to_isaaclab.append(il_idx)

print("=" * 80)
print("G1 Joint Order Mapping: MuJoCo/AMASS → IsaacLab (BFS)")
print("=" * 80)
print("\nMapping array (use this in MotionLoader):")
print(f"MUJOCO_TO_ISAACLAB_G1 = {mujoco_to_isaaclab}")
print("\nVerification:")
for i, (mj_name, il_idx) in enumerate(zip(MUJOCO_ORDER, mujoco_to_isaaclab)):
    il_name = ISAACLAB_ORDER[il_idx]
    check = "✓" if mj_name == il_name else "✗ MISMATCH"
    print(f"  MJ[{i:2d}] → IL[{il_idx:2d}]  {mj_name:30s} {check}")
