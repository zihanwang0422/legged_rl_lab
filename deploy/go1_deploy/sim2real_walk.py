#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Sim2Real deployment for Unitree Go1.

Reads joystick commands from the Go1 wireless remote (LowState.wirelessRemote),
runs the JIT-traced policy, and sends position commands via unitree_legged_sdk.

Usage (on the Go1 onboard computer or a wired PC):
    python deploy/go1_deploy/sim2real_walk.py --model go1_flat.pt --config go1_walk.yaml

Prerequisites:
    1. Build unitree_legged_sdk (see deploy/go1_deploy/unitree_legged_sdk/README.md)
    2. Connect to Go1 via Ethernet (192.168.123.x subnet)
    3. Put Go1 into low-level mode (L2+A, L2+A, L1+L2+Start)
"""

import sys
import os
import time
import math
import struct
import argparse
import numpy as np
import torch
import yaml
from types import SimpleNamespace
from collections import deque

# Add local unitree_legged_sdk to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "unitree_legged_sdk", "lib", "python", "amd64"))
import robot_interface as sdk


# =====================================================================
# Observation builder (must match sim2sim_walk.py / training order)
# =====================================================================

def build_obs(base_ang_vel, projected_gravity, commands,
              dof_pos_rel, dof_vel, last_action, config):
    """48-dim single-frame observation (Go1 12 DOF)."""
    obs = np.empty(9 + 3 * 12, dtype=np.float32)
    obs[0:3] = base_ang_vel * config.obs_scales["base_ang_vel"]
    obs[3:6] = projected_gravity
    obs[6:9] = commands
    obs[9:21] = dof_pos_rel * config.obs_scales["joint_pos"]
    obs[21:33] = dof_vel * config.obs_scales["joint_vel"]
    obs[33:45] = last_action
    return obs


# =====================================================================
# Sim2Real Controller
# =====================================================================

class Go1RealController:
    """Low-level position controller for Go1 via unitree_legged_sdk."""

    # SDK motor indices (FR, FL, RR, RL — Unitree convention)
    SDK_IDX = {
        "FR_0": 0, "FR_1": 1, "FR_2": 2,
        "FL_0": 3, "FL_1": 4, "FL_2": 5,
        "RR_0": 6, "RR_1": 7, "RR_2": 8,
        "RL_0": 9, "RL_1": 10, "RL_2": 11,
    }

    # Isaac order → SDK motor name mapping (12 joints)
    #  Isaac: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, ...
    #  SDK:   FL_0,   FR_0,   RL_0,   RR_0,   FL_1,     FR_1,     ...
    ISAAC_TO_SDK = [
        "FL_0", "FR_0", "RL_0", "RR_0",   # hips
        "FL_1", "FR_1", "RL_1", "RR_1",   # thighs
        "FL_2", "FR_2", "RL_2", "RR_2",   # calves
    ]

    # SDK → Isaac order for reading sensor data
    # SDK order: FR0 FR1 FR2 FL0 FL1 FL2 RR0 RR1 RR2 RL0 RL1 RL2
    # We read in Isaac order:
    SDK_TO_ISAAC = [
        ("FL_0", 0), ("FR_0", 1), ("RL_0", 2), ("RR_0", 3),
        ("FL_1", 4), ("FR_1", 5), ("RL_1", 6), ("RR_1", 7),
        ("FL_2", 8), ("FR_2", 9), ("RL_2", 10), ("RR_2", 11),
    ]

    def __init__(self, config_path: str, model_name: str):
        # ---- Load config ----
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        self.config = cfg = SimpleNamespace(**raw)
        for k, v in raw.items():
            if isinstance(v, list) and k not in (
                "joint_names_mujoco", "actuator_names_mujoco", "sdk_joint_order",
            ):
                try:
                    v = np.array(v, dtype=np.int32 if "map" in k else np.float32)
                except (ValueError, TypeError):
                    pass
            setattr(cfg, k, v)

        # ---- Load policy ----
        policy_path = os.path.join(_SCRIPT_DIR, "exported_policy", model_name)
        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")

        # ---- Config shortcuts ----
        self.dt = cfg.policy_dt  # 0.02 s (50 Hz)
        self.num_joints = 12
        self.default_qpos_isaac = cfg.default_qpos_isaac  # (12,) np array
        self.action_scale = cfg.action_scale["pos"]

        # PD gains — expand scalar to per-joint array if needed
        self.kp = np.broadcast_to(np.asarray(cfg.kp_walk, dtype=np.float32), (12,)).copy()
        self.kd = np.broadcast_to(np.asarray(cfg.kd_walk, dtype=np.float32), (12,)).copy()

        # ---- SDK setup ----
        robot_ip = getattr(cfg, "robot_ip", "192.168.123.10")
        robot_port = getattr(cfg, "robot_port", 8007)
        local_port = getattr(cfg, "local_port", 8080)
        print(f"Connecting to Go1 @ {robot_ip}:{robot_port} (local {local_port})")

        self.udp = sdk.UDP(0xFF, local_port, robot_ip, robot_port)
        self.safe = sdk.Safety(sdk.LeggedType.Go1)
        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)

        # ---- Buffers ----
        self.last_action = np.zeros(12, dtype=np.float32)
        self.obs_history = deque(
            [np.zeros(9 + 3 * 12, dtype=np.float32)] * 5, maxlen=5
        )

        # Gyro smoothing
        self.body_ang_vel = np.zeros(3, dtype=np.float64)
        self.smoothing_ratio = 0.2

        print("Go1 real controller initialised.")

    # ----------------------------------------------------------------
    # Sensor reading helpers
    # ----------------------------------------------------------------

    def _recv(self):
        self.udp.Recv()
        self.udp.GetRecv(self.state)

    def _get_joint_pos_isaac(self) -> np.ndarray:
        """Return 12-dim joint positions in Isaac order."""
        pos = np.empty(12, dtype=np.float32)
        for sdk_name, isaac_idx in self.SDK_TO_ISAAC:
            pos[isaac_idx] = self.state.motorState[self.SDK_IDX[sdk_name]].q
        return pos

    def _get_joint_vel_isaac(self) -> np.ndarray:
        """Return 12-dim joint velocities in Isaac order."""
        vel = np.empty(12, dtype=np.float32)
        for sdk_name, isaac_idx in self.SDK_TO_ISAAC:
            vel[isaac_idx] = self.state.motorState[self.SDK_IDX[sdk_name]].dq
        return vel

    def _get_body_ang_vel(self) -> np.ndarray:
        """Smoothed body angular velocity from IMU gyroscope."""
        gyro = np.array(self.state.imu.gyroscope, dtype=np.float64)
        self.body_ang_vel = (
            self.smoothing_ratio * gyro
            + (1.0 - self.smoothing_ratio) * self.body_ang_vel
        )
        return self.body_ang_vel.astype(np.float32)

    def _get_projected_gravity(self) -> np.ndarray:
        """Projected gravity from IMU roll/pitch."""
        roll, pitch = self.state.imu.rpy[0], self.state.imu.rpy[1]
        # Rotation from world to body (small-angle Rodrigues isn't used;
        # we directly compute the gravity projection)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        # g_body = R_body_world @ [0, 0, -1]
        gx = sp
        gy = -sr * cp
        gz = -cr * cp
        return np.array([gx, gy, gz], dtype=np.float32)

    def _get_commands(self) -> np.ndarray:
        """Read velocity commands from Go1 wireless remote."""
        wr = self.state.wirelessRemote
        ly = struct.unpack("f", struct.pack("4B", *wr[20:24]))[0]  # forward
        lx = struct.unpack("f", struct.pack("4B", *wr[4:8]))[0]    # lateral
        rx = struct.unpack("f", struct.pack("4B", *wr[8:12]))[0]   # yaw

        # Scale & deadzone
        forward = ly * 0.6
        if abs(forward) < 0.15:
            forward = 0.0
        side = -lx * 0.5
        if abs(side) < 0.15:
            side = 0.0
        rotate = -rx * 0.8
        if abs(rotate) < 0.2:
            rotate = 0.0

        return np.array([forward, side, rotate], dtype=np.float32)

    # ----------------------------------------------------------------
    # Motor command helpers
    # ----------------------------------------------------------------

    def _send_joint_pos(self, target_isaac: np.ndarray, kp, kd):
        """Send position commands in Isaac order to SDK motors."""
        for isaac_idx, sdk_name in enumerate(self.ISAAC_TO_SDK):
            motor_id = self.SDK_IDX[sdk_name]
            self.cmd.motorCmd[motor_id].q = float(target_isaac[isaac_idx])
            self.cmd.motorCmd[motor_id].dq = 0.0
            self.cmd.motorCmd[motor_id].Kp = float(
                kp[isaac_idx] if hasattr(kp, "__len__") else kp
            )
            self.cmd.motorCmd[motor_id].Kd = float(
                kd[isaac_idx] if hasattr(kd, "__len__") else kd
            )
            self.cmd.motorCmd[motor_id].tau = 0.0

    def _safe_send(self):
        self.safe.PowerProtect(self.cmd, self.state, 9)
        self.udp.SetSend(self.cmd)
        self.udp.Send()

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------

    def run(self):
        """Stand up → stabilise → policy loop."""
        cfg = self.config
        standup_steps = int(getattr(cfg, "standup_duration", 2.0) / self.dt)
        stabilize_steps = int(getattr(cfg, "stabilize_duration", 0.5) / self.dt)
        warmup_steps = 80  # ~1.6 s at 50 Hz, zero commands / zero vel

        print("\n" + "=" * 60)
        print("  Go1 Sim2Real — Standing up …")
        print("=" * 60)

        step = 0
        phase = "standup"  # standup → stabilize → warmup → run

        try:
            while True:
                t0 = time.time()
                self._recv()

                # ---- Phase: Stand-up (soft → stiff) ----
                if phase == "standup":
                    kp = 5.0 if step < 50 else 50.0
                    kd = 1.0 if step < 50 else 5.0
                    self._send_joint_pos(self.default_qpos_isaac, kp, kd)
                    self._safe_send()
                    step += 1
                    if step >= standup_steps:
                        step = 0
                        phase = "stabilize"
                        print("  Stabilising …")

                # ---- Phase: Stabilise (hold default pose with deploy gains) ----
                elif phase == "stabilize":
                    self._send_joint_pos(self.default_qpos_isaac, self.kp, self.kd)
                    self._safe_send()
                    step += 1
                    if step >= stabilize_steps:
                        step = 0
                        phase = "warmup"
                        print("  Warming up sensors …")

                # ---- Phase: Warmup (run policy with zero commands) ----
                elif phase == "warmup":
                    obs = self._build_obs_step(force_zero_cmd=True, force_zero_vel=(step < 40))
                    action = self._infer(obs)
                    target = action * self.action_scale + self.default_qpos_isaac
                    self._send_joint_pos(target, self.kp, self.kd)
                    self._safe_send()
                    step += 1
                    if step >= warmup_steps:
                        phase = "run"
                        print("\n  ✅ Policy running! Use wireless remote to control.")
                        print("     Ly=forward  Lx=lateral  Rx=yaw\n")

                # ---- Phase: Run ----
                else:
                    obs = self._build_obs_step(force_zero_cmd=False, force_zero_vel=False)
                    action = self._infer(obs)
                    target = action * self.action_scale + self.default_qpos_isaac
                    self._send_joint_pos(target, self.kp, self.kd)
                    self._safe_send()
                    step += 1

                    if step % 50 == 0:
                        cmd = self._get_commands()
                        print(
                            f"[{step:6d}] cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})"
                            f"  freq={1.0 / max(time.time() - t0, 1e-6):.1f} Hz"
                        )

                # ---- Timing ----
                elapsed = time.time() - t0
                sleep_t = self.dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n  Interrupted — holding default pose …")
            for _ in range(200):
                self._recv()
                self._send_joint_pos(self.default_qpos_isaac, 30.0, 5.0)
                self._safe_send()
                time.sleep(self.dt)
            print("  Done.")

    # ----------------------------------------------------------------
    # Observation / inference helpers
    # ----------------------------------------------------------------

    def _build_obs_step(self, force_zero_cmd: bool, force_zero_vel: bool) -> np.ndarray:
        """Build group-major history observation (5 frames, 270-dim)."""
        ang_vel = self._get_body_ang_vel()
        proj_grav = self._get_projected_gravity()
        joint_pos = self._get_joint_pos_isaac()
        joint_vel = self._get_joint_vel_isaac()

        if force_zero_vel:
            ang_vel = np.zeros(3, dtype=np.float32)
            joint_vel = np.zeros(12, dtype=np.float32)

        cmd = np.zeros(3, dtype=np.float32) if force_zero_cmd else self._get_commands()
        dof_pos_rel = joint_pos - self.default_qpos_isaac

        obs_frame = build_obs(ang_vel, proj_grav, cmd, dof_pos_rel, joint_vel,
                              self.last_action, self.config)
        self.obs_history.append(obs_frame)

        # Group-major reorder: [omega×5, grav×5, cmd×5, pos×5, vel×5, act×5]
        obs_arr = np.array(list(self.obs_history))  # (5, 45)
        n = 12
        obs_input = np.concatenate([
            obs_arr[:, 0:3].reshape(-1),
            obs_arr[:, 3:6].reshape(-1),
            obs_arr[:, 6:9].reshape(-1),
            obs_arr[:, 9:9 + n].reshape(-1),
            obs_arr[:, 9 + n:9 + 2 * n].reshape(-1),
            obs_arr[:, 9 + 2 * n:9 + 3 * n].reshape(-1),
        ])
        return obs_input

    def _infer(self, obs_input: np.ndarray) -> np.ndarray:
        """Run JIT policy and return Isaac-order action (12-dim)."""
        obs_batch = obs_input[np.newaxis, :].astype(np.float32)
        with torch.no_grad():
            out = self.policy(torch.from_numpy(obs_batch))
            if isinstance(out, tuple):
                out = out[0]
            action = out.cpu().numpy().flatten().astype(np.float32)
        self.last_action = action.copy()
        return action


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go1 Sim2Real Walk")
    parser.add_argument("--model", type=str, default="go1_flat.pt",
                        help="Policy file in exported_policy/")
    parser.add_argument("--config", type=str, default="go1_walk.yaml",
                        help="Config YAML in config/")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(_SCRIPT_DIR, "config", args.config)

    ctrl = Go1RealController(config_path, args.model)
    ctrl.run()
