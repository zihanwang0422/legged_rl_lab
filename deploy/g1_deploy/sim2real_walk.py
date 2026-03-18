#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Sim2Real deployment for Unitree G1 (Unitree SDK2 Python, low-level).

This script mirrors `deploy/g1_deploy/sim2sim_walk.py` observation/action interface:
- 29 DOF, Isaac training order
- 5-frame history, group-major flattening
- policy_dt inference, sim_dt continuous command publishing

Usage:
  python deploy/g1_deploy/sim2real_walk.py --model g1_walk.pt --config g1_walk.yaml --iface enp108s0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R


# ======== Paths / Imports ========

THIS_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = THIS_DIR.parent


def _bootstrap_cyclonedds_home() -> None:
    """Ensure CYCLONEDDS_HOME points to a valid local install before SDK import."""
    env_home = os.environ.get("CYCLONEDDS_HOME", "")
    if env_home and (Path(env_home) / "lib" / "libddsc.so").exists():
        return

    candidates = [
        THIS_DIR / "cyclonedds" / "install",
        DEPLOY_DIR / "cyclonedds" / "install",
    ]
    for candidate in candidates:
        if (candidate / "lib" / "libddsc.so").exists():
            os.environ["CYCLONEDDS_HOME"] = str(candidate)
            print(f"[CycloneDDS] Using CYCLONEDDS_HOME={candidate}")
            return


_bootstrap_cyclonedds_home()
from common.remote_controller import KeyMap, RemoteController

# Unitree SDK2 python (vendored under deploy/g1_deploy/unitree_sdk2_python/)
SDK2_PY_ROOT = THIS_DIR / "unitree_sdk2_python"
sys.path.insert(0, str(SDK2_PY_ROOT))

from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (  # noqa: E402
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HGLowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HGLowState_  # noqa: E402
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402
from unitree_sdk2py.utils.thread import RecurrentThread  # noqa: E402


# ======== Math helpers (same as sim2sim) ========

def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector world->body using inverse quaternion.

    q is [x, y, z, w].
    """
    q_w = float(q_xyzw[3])
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * float(np.dot(q_vec, v)) * 2.0
    return a - b + c


def compute_projected_gravity(q_xyzw: np.ndarray) -> np.ndarray:
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return quat_rotate_inverse(q_xyzw, gravity_world).astype(np.float32)


def build_obs(
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    commands: np.ndarray,
    dof_pos_rel: np.ndarray,
    dof_vel: np.ndarray,
    last_action: np.ndarray,
    config: SimpleNamespace,
) -> np.ndarray:
    """Build 96D obs for G1 29DOF velocity policy (single frame)."""
    obs = []
    obs.extend(list(base_ang_vel * config.obs_scales["base_ang_vel"]))
    obs.extend(list(projected_gravity))
    obs.extend(list(commands))
    obs.extend(list(dof_pos_rel * config.obs_scales["joint_pos"]))
    obs.extend(list(dof_vel * config.obs_scales["joint_vel"]))
    obs.extend(list(last_action))
    return np.asarray(obs, dtype=np.float32)


def apply_deadzone(value: float, deadzone: float = 0.08) -> float:
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def scale_axis(value: float, vmin: float, vmax: float) -> float:
    if value >= 0.0:
        return value * float(vmax)
    return value * abs(float(vmin))


# ======== Config loader (aligned with sim2sim_walk.py) ========

_STRING_LIST_KEYS = {"joint_names_mujoco", "actuator_names_mujoco", "sdk_joint_order"}


def _resolve_path(base_dir: Path, p: str) -> str:
    if not isinstance(p, str) or p == "":
        return p
    if os.path.isabs(p):
        return p
    # support yaml values like "./exported_policy"
    return str((base_dir / p).resolve())


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    cfg = SimpleNamespace(**raw_cfg)
    for k, v in raw_cfg.items():
        if isinstance(v, list):
            if k in _STRING_LIST_KEYS:
                setattr(cfg, k, v)
                continue
            try:
                v = np.array(v, dtype=np.int32 if "map" in k else np.float32)
            except (ValueError, TypeError):
                pass
        setattr(cfg, k, v)

    # Resolve key paths relative to this deploy folder (same convention as sim2sim)
    cfg.policy_path = _resolve_path(THIS_DIR, str(getattr(cfg, "policy_path", "./exported_policy")))
    cfg.xml_path = _resolve_path(THIS_DIR, str(getattr(cfg, "xml_path", "./assets/scene_terrain.xml")))
    return cfg


# ======== Real robot controller ========

class G1Sim2RealController:
    def __init__(self, config: SimpleNamespace, model_name: str, iface: str, dry_run: bool = False):
        self.config = config
        self.iface = iface
        self.dry_run = dry_run

        self.num_joints = len(config.sdk_joint_order)
        if self.num_joints != 29:
            raise ValueError(f"Expected 29 joints, got {self.num_joints}. Check `sdk_joint_order` in yaml.")

        # Policy
        policy_path = str((Path(config.policy_path) / model_name).resolve())
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy not found: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        # Buffers
        self.history_length = config.history_length
        self.obs_history = deque([np.zeros(96, dtype=np.float32)] * self.history_length, maxlen=self.history_length)
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)

        # Default pose (Isaac order) and mapping assumption:
        # sdk_joint_order is configured to match MuJoCo joint order in `joint_names_mujoco`.
        self.default_qpos_isaac = np.asarray(config.default_qpos_isaac, dtype=np.float32)
        self.mujoco_to_isaac = np.asarray(config.mujoco_to_isaac_map, dtype=np.int32)
        self.isaac_to_mujoco = np.asarray(config.isaac_to_mujoco_map, dtype=np.int32)

        # Timing
        self.sim_dt = float(config.sim_dt)  # 200 Hz loop
        self.policy_dt = float(config.policy_dt)  # e.g. 0.02 -> 50 Hz
        self.policy_decimation = int(round(self.policy_dt / self.sim_dt))
        self.policy_counter = 0

        # DDS topics (match official low-level examples)
        self.lowstate_topic = str(getattr(config, "dds_lowstate_topic", "rt/lowstate"))
        self.lowcmd_topic = str(getattr(config, "dds_lowcmd_topic", "rt/lowcmd"))
        self.mode_pr = 0
        self.mode_machine = 0

        # SDK2 comms
        ChannelFactoryInitialize(0, iface)
        self._release_motion_mode()

        self.pub = ChannelPublisher(self.lowcmd_topic, HGLowCmd_)
        self.sub = ChannelSubscriber(self.lowstate_topic, HGLowState_)
        self.pub.Init()

        self._latest_state: Optional[HGLowState_] = None
        self._state_recv_ts: float = 0.0

        def _handler(msg: HGLowState_):
            self._latest_state = msg
            self._state_recv_ts = time.time()
            self.mode_machine = int(msg.mode_machine)
            self.remote_controller.set(msg.wireless_remote)

        # Queue depth 10 is enough for 200Hz state.
        self.sub.Init(_handler, 10)

        self.crc = CRC()

        # Cached command, asynchronous publisher and watchdog.
        self.cmd_lock = Lock()
        self._cached_cmd: Optional[unitree_hg_msg_dds__LowCmd_] = None
        self._cmd_update_ts: float = 0.0
        self._watchdog_timeout_s = float(getattr(config, "cmd_watchdog_timeout_s", 0.08))
        self._publish_dt = float(getattr(config, "publish_dt", 1.0 / 500.0))
        self._watchdog_triggered = False
        self._publish_thread_started = False
        self._publish_thread = RecurrentThread(interval=self._publish_dt, target=self._publish_with_watchdog)
        self._active_policy_idx = 1
        self.remote_controller = RemoteController()

        print(f"[DDS] lowstate topic: {self.lowstate_topic}")
        print(f"[DDS] lowcmd topic:   {self.lowcmd_topic}")
        print(f"[CTRL] publish_dt={self._publish_dt:.4f}s, watchdog={self._watchdog_timeout_s:.3f}s")

    def _release_motion_mode(self):
        """Release any high-level motion service that blocks low-level control."""
        try:
            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()
            status, result = msc.CheckMode()
            retry = 0
            while result.get("name") and retry < 10:
                print(f"[MotionSwitcher] Releasing mode: {result.get('name')}")
                msc.ReleaseMode()
                time.sleep(0.3)
                status, result = msc.CheckMode()
                retry += 1
            if result.get("name"):
                print("[MotionSwitcher] Warning: mode is still active, low-level command may be rejected.")
        except Exception as e:
            print(f"[MotionSwitcher] Warning: {e}")

    def _get_state_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        msg = self._latest_state
        if msg is None:
            raise RuntimeError("No lowstate received yet.")

        # base angular velocity (body frame)
        base_ang_vel = np.asarray(msg.imu_state.gyroscope, dtype=np.float32)

        # projected gravity using quaternion if available
        # IMU quaternion in Unitree messages is typically [w, x, y, z]
        q_wxyz = np.asarray(msg.imu_state.quaternion, dtype=np.float32)
        if np.linalg.norm(q_wxyz) > 1e-6:
            q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
        else:
            # fallback: build from rpy
            rpy = np.asarray(msg.imu_state.rpy, dtype=np.float32)
            q_xyzw = R.from_euler("xyz", rpy).as_quat().astype(np.float32)
        projected_gravity = compute_projected_gravity(q_xyzw)

        # joints (SDK order -> Isaac order)
        q_sdk = np.array([msg.motor_state[i].q for i in range(self.num_joints)], dtype=np.float32)
        dq_sdk = np.array([msg.motor_state[i].dq for i in range(self.num_joints)], dtype=np.float32)

        q_isaac = q_sdk[self.mujoco_to_isaac]
        dq_isaac = dq_sdk[self.mujoco_to_isaac]

        return base_ang_vel, projected_gravity, q_isaac, dq_isaac

    def _load_policy(self, config_path: str, model_name: str, policy_idx: int):
        """Hot-swap config + policy (mirrors sim2sim load_policy)."""
        print(f"[PolicySwitch] Loading policy {policy_idx}: {config_path} / {model_name}")
        new_cfg = load_config(config_path)
        policy_path = str((Path(new_cfg.policy_path) / model_name).resolve())
        if not os.path.exists(policy_path):
            print(f"[PolicySwitch] Policy file not found: {policy_path}, skipping.")
            return
        self.config = new_cfg
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()
        self.policy_decimation = int(round(new_cfg.policy_dt / new_cfg.sim_dt))
        self.default_qpos_isaac = np.asarray(new_cfg.default_qpos_isaac, dtype=np.float32)
        self.mujoco_to_isaac = np.asarray(new_cfg.mujoco_to_isaac_map, dtype=np.int32)
        self.isaac_to_mujoco = np.asarray(new_cfg.isaac_to_mujoco_map, dtype=np.int32)
        self.history_length = new_cfg.history_length
        self.obs_history = deque([np.zeros(96, dtype=np.float32)] * self.history_length, maxlen=self.history_length)
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self._active_policy_idx = policy_idx
        print(f"[PolicySwitch] Switched to policy {policy_idx}")

    def _make_lowcmd(self, target_qpos_sdk: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> unitree_hg_msg_dds__LowCmd_:
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_pr = self.mode_pr
        cmd.mode_machine = self.mode_machine

        # Leave mode fields default (0). Most firmware uses motor_cmd[].mode to enable servo.
        # Set only the first 29 motors (matching sdk_joint_order).
        for i in range(35):
            cmd.motor_cmd[i].mode = 0x00
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0

        for i in range(self.num_joints):
            cmd.motor_cmd[i].mode = 0x01  # position/servo enable
            cmd.motor_cmd[i].q = float(target_qpos_sdk[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = float(kp[i])
            cmd.motor_cmd[i].kd = float(kd[i])

        cmd.crc = self.crc.Crc(cmd)
        return cmd

    def _make_damping_cmd(self) -> unitree_hg_msg_dds__LowCmd_:
        """Create a damping (zero torque) command (all gains=0, mode=0)."""
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_pr = self.mode_pr
        cmd.mode_machine = self.mode_machine
        for i in range(35):
            cmd.motor_cmd[i].mode = 0x00  # Shutdown all motors
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
        cmd.crc = self.crc.Crc(cmd)
        return cmd

    def _set_cached_cmd(self, cmd: unitree_hg_msg_dds__LowCmd_):
        with self.cmd_lock:
            self._cached_cmd = cmd
            self._cmd_update_ts = time.time()

    def _publish_with_watchdog(self):
        if self.dry_run:
            return

        now = time.time()
        with self.cmd_lock:
            cmd = self._cached_cmd
            cmd_update_ts = self._cmd_update_ts

        if cmd is None:
            return

        stale_cmd = cmd_update_ts > 0.0 and (now - cmd_update_ts) > self._watchdog_timeout_s
        stale_state = self._state_recv_ts > 0.0 and (now - self._state_recv_ts) > 0.2

        if stale_cmd or stale_state:
            if not self._watchdog_triggered:
                print(
                    f"[Watchdog] Triggered: stale_cmd={stale_cmd}, stale_state={stale_state}. Sending damping cmd."
                )
                self._watchdog_triggered = True
            self.pub.Write(self._make_damping_cmd())
            return

        if self._watchdog_triggered:
            print("[Watchdog] Recovered, resume lowcmd publishing.")
            self._watchdog_triggered = False

        self.pub.Write(cmd)

    def _publish_cached(self):
        if self.dry_run:
            return
        with self.cmd_lock:
            if self._cached_cmd is None:
                return
            self.pub.Write(self._cached_cmd)

    def run(self, policy_registry=None):
        c = self.config
        if policy_registry is None:
            policy_registry = {}

        # Wait for first valid state
        print(f"等待 `rt/lowstate` 数据... (iface={self.iface})")
        t0 = time.time()
        while self._latest_state is None:
            if time.time() - t0 > 5.0:
                raise TimeoutError(
                    "5 秒内未收到 lowstate。请检查：机器人已进入 Low Level、网卡 iface 正确、PC 与机器人在同网段。"
                )
            time.sleep(0.01)

        print("\n✓ 已连接到机器人")
        print("=" * 70)
        print("按下 Start 键 → 默认位置状态")
        print("在默认位置状态下，按下 A 键 → 运动控制模式")
        print("在运动控制模式下，按下 Start 键 → 阻尼模式（退出）")
        print("=" * 70 + "\n")

        # State machine: 0=零力矩(waiting), 1=默认位置, 2=运动控制, 3=阻尼(damping)
        state = 0  # Start in zero-torque state
        state_names = {
            0: "零力矩状态",
            1: "默认位置状态",
            2: "运动控制模式",
            3: "阻尼模式",
        }

        # Initialize command to zero (zero-torque)
        kp = np.asarray(c.kp_walk, dtype=np.float32)
        kd = np.asarray(c.kd_walk, dtype=np.float32)
        target_qpos_isaac = self.default_qpos_isaac.copy()
        target_qpos_sdk = target_qpos_isaac[self.isaac_to_mujoco]
        self._set_cached_cmd(self._make_lowcmd(target_qpos_sdk, kp, kd))

        if not self._publish_thread_started:
            self._publish_thread.Start()
            self._publish_thread_started = True

        # Main loop (200 Hz)
        start_time = time.time()
        motion_steps = 0
        deadzone = float(getattr(c, "gamepad_deadzone", 0.05))
        prev_buttons = [0] * 16

        try:
            while True:
                loop_t0 = time.time()

                btn = self.remote_controller.button
                start_edge = btn[KeyMap.start] == 1 and prev_buttons[KeyMap.start] == 0
                a_edge = btn[KeyMap.A] == 1 and prev_buttons[KeyMap.A] == 0
                select_edge = btn[KeyMap.select] == 1 and prev_buttons[KeyMap.select] == 0

                # R1 is treated as RB for policy switch combos.
                rb_hold = btn[KeyMap.R1] == 1
                b_edge = btn[KeyMap.B] == 1 and prev_buttons[KeyMap.B] == 0
                x_edge = btn[KeyMap.X] == 1 and prev_buttons[KeyMap.X] == 0
                y_edge = btn[KeyMap.Y] == 1 and prev_buttons[KeyMap.Y] == 0

                if start_edge and state == 0:
                    print(f"\n[状态转移] {state_names[state]} → 默认位置状态")
                    print("机器人移动到默认位置。准备好后，缓慢降低吊装。")
                    state = 1
                elif start_edge and state in [1, 2]:
                    print(f"\n[状态转移] {state_names[state]} → 阻尼模式")
                    state = 3

                if select_edge and state != 3:
                    print(f"\n[状态转移] {state_names[state]} → 阻尼模式 (Select)")
                    state = 3

                # Basic stale-state protection
                if (time.time() - self._state_recv_ts) > 0.2:
                    print("\n⚠️  lowstate 超时(>0.2s)，停止发送 lowcmd。")
                    self._set_cached_cmd(self._make_damping_cmd())
                    self._publish_cached()
                    break

                # Read current state arrays
                base_ang_vel, projected_gravity, q_isaac, dq_isaac = self._get_state_arrays()

                # Simple tilt protection using IMU rpy
                rpy = np.asarray(self._latest_state.imu_state.rpy, dtype=np.float32)
                if abs(float(rpy[0])) > 0.8 or abs(float(rpy[1])) > 0.8:
                    print(f"\n⚠️  倾倒保护触发: roll={rpy[0]:+.2f}, pitch={rpy[1]:+.2f}，进入阻尼模式。")
                    if state != 3:
                        state = 3

                # Get current velocity commands from official wireless_remote.
                lx = apply_deadzone(float(self.remote_controller.lx), deadzone)
                ly = apply_deadzone(float(self.remote_controller.ly), deadzone)
                rx = apply_deadzone(float(self.remote_controller.rx), deadzone)
                cmd_vx = scale_axis(ly, c.vx_range[0], c.vx_range[1])
                cmd_vy = scale_axis(-lx, c.vy_range[0], c.vy_range[1])
                cmd_vyaw = scale_axis(-rx, c.vyaw_range[0], c.vyaw_range[1])

                if a_edge and state == 1:
                    print(f"\n[状态转移] {state_names[state]} → 运动控制模式")
                    print("开始策略推理。逐渐降低吊绳。")
                    state = 2

                current_policy_idx = self._active_policy_idx
                if rb_hold and b_edge:
                    current_policy_idx = 2
                elif rb_hold and x_edge:
                    current_policy_idx = 3
                elif rb_hold and y_edge:
                    current_policy_idx = 4

                # 策略切换检测 (对齐 sim2sim，仅切换 policy 2,3,4)
                if current_policy_idx > 1 and current_policy_idx != getattr(self, '_active_policy_idx', 1):
                    if current_policy_idx in policy_registry:
                        cfg_path, mdl_name = policy_registry[current_policy_idx]
                        self._load_policy(cfg_path, mdl_name, current_policy_idx)
                        c = self.config
                        kp = np.asarray(c.kp_walk, dtype=np.float32)
                        kd = np.asarray(c.kd_walk, dtype=np.float32)

                # State machine execution
                if state == 0:
                    # 零力矩状态：hold default position with minimal gains
                    target_qpos_isaac = self.default_qpos_isaac.copy()
                    target_qpos_sdk = target_qpos_isaac[self.isaac_to_mujoco]
                    kp_hold = np.asarray(c.kp_walk, dtype=np.float32) * 0.0  # Zero gains
                    kd_hold = np.asarray(c.kd_walk, dtype=np.float32) * 0.0
                    self._set_cached_cmd(self._make_lowcmd(target_qpos_sdk, kp_hold, kd_hold))

                elif state == 1:
                    # 默认位置状态：hold default position with medium gains
                    target_qpos_isaac = self.default_qpos_isaac.copy()
                    target_qpos_sdk = target_qpos_isaac[self.isaac_to_mujoco]
                    kp_default = np.asarray(c.kp_walk, dtype=np.float32) * 0.3  # Reduced gains
                    kd_default = np.asarray(c.kd_walk, dtype=np.float32) * 0.3
                    self._set_cached_cmd(self._make_lowcmd(target_qpos_sdk, kp_default, kd_default))

                elif state == 2:
                    # 运动控制模式：run policy
                    self.policy_counter += 1
                    if self.policy_counter >= self.policy_decimation:
                        self.policy_counter = 0

                        commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)

                        dof_pos_rel = q_isaac - self.default_qpos_isaac
                        obs = build_obs(
                            base_ang_vel=base_ang_vel,
                            projected_gravity=projected_gravity,
                            commands=commands,
                            dof_pos_rel=dof_pos_rel,
                            dof_vel=dq_isaac,
                            last_action=self.last_action,
                            config=c,
                        )
                        self.obs_history.append(obs)

                        # Group-major flattening (same as sim2sim_walk.py)
                        obs_arr = np.asarray(list(self.obs_history), dtype=np.float32)  # (5, 96)
                        n = self.num_joints  # 29
                        obs_input = np.concatenate(
                            [
                                obs_arr[:, 0:3].reshape(-1),
                                obs_arr[:, 3:6].reshape(-1),
                                obs_arr[:, 6:9].reshape(-1),
                                obs_arr[:, 9 : 9 + n].reshape(-1),
                                obs_arr[:, 9 + n : 9 + 2 * n].reshape(-1),
                                obs_arr[:, 9 + 2 * n : 9 + 3 * n].reshape(-1),
                            ],
                            axis=0,
                        )
                        obs_batch = obs_input[np.newaxis, :].astype(np.float32)

                        with torch.no_grad():
                            action_tensor = self.policy(torch.from_numpy(obs_batch))
                            if isinstance(action_tensor, tuple):
                                action_tensor = action_tensor[0]
                            action = action_tensor.cpu().numpy().reshape(-1).astype(np.float32)

                        if action.shape[0] < n:
                            raise RuntimeError(f"Policy output dim {action.shape[0]} < {n} (expected 29).")

                        self.last_action = action[:n].copy()

                        target_qpos_isaac = self.last_action * float(c.action_scale["pos"]) + self.default_qpos_isaac
                        target_qpos_sdk = target_qpos_isaac[self.isaac_to_mujoco]

                        self._set_cached_cmd(self._make_lowcmd(target_qpos_sdk, kp, kd))

                elif state == 3:
                    # 阻尼模式：zero torque and exit
                    self._set_cached_cmd(self._make_damping_cmd())
                    # Publish one damping command and exit
                    self._publish_cached()
                    print("已发送阻尼命令，程序退出。")
                    break

                motion_steps += 1
                if motion_steps % int(1.0 / self.sim_dt) == 0:
                    t_real = time.time() - start_time
                    hz = motion_steps / t_real if t_real > 1e-6 else 0.0
                    print(
                        f"[{state_names[state]}] vx={cmd_vx:+.2f} | vy={cmd_vy:+.2f} | yaw={cmd_vyaw:+.2f}  "
                        f"t={t_real:.1f}s, Hz={hz:.1f}"
                    )

                # Timing
                elapsed = time.time() - loop_t0
                sleep_t = self.sim_dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

                prev_buttons = btn.copy()
        finally:
            if self._publish_thread_started:
                self._publish_thread.Wait()


def _resolve_config_path(name: str) -> str:
    if os.path.exists(name):
        return name
    return str((THIS_DIR / "config" / name).resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="g1_walk.pt")
    parser.add_argument("--config", type=str, default="g1_walk.yaml")
    parser.add_argument(
        "--iface",
        type=str,
        default=os.environ.get("UNITREE_IFACE", "enp108s0"),
        help="DDS network interface name, e.g. enp2s0/eth0",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only run policy loop, do not publish lowcmd.")
    args = parser.parse_args()

    cfg_path = _resolve_config_path(args.config)
    cfg = load_config(cfg_path)

    def resolve_config(name):
        if os.path.exists(name):
            return name
        return str((THIS_DIR / "config" / name).resolve())

    # 策略注册表 (对齐 sim2sim)
    policy_registry = {
        1: (cfg_path,                              args.model),
        2: (resolve_config('policy1.yaml'),        'policy1.pt'),
        3: (resolve_config('policy2.yaml'),        'policy2.pt'),
        4: (resolve_config('policy3.yaml'),        'policy3.pt'),
    }

    print("\n" + "=" * 70)
    print(f"Unitree G1 Sim2Real (SDK2) | iface={args.iface} | dry_run={args.dry_run}")
    print("Input: RemoteController (from LowState.wireless_remote)")
    print("=" * 70)
    print("前置条件：机器人已进入 Low Level / Debug 模式，并允许外部低层控制。")
    print("\n状态流程：")
    print("  1. 初始化 → 零力矩状态（等待输入）")
    print("  2. 按 Start 键 → 默认位置状态（移动到默认姿态）")
    print("  3. 按 A 键 → 运动控制模式（执行策略）")
    print("  4. 按 Start 键 → 阻尼模式（释放扭矩，程序退出）")
    print("\n控制映射（运动控制模式）：")
    print("  左摇杆 上/下  : vx (前进/后退)")
    print("  左摇杆 左/右  : vy (横移)")
    print("  右摇杆 左/右  : vyaw (转向)")
    print("  A            : 默认位置 -> 运动控制")
    print("  RB + B/X/Y   : 策略切换 1/2/3（占位符）")
    print("\n安全保护：")
    print("  • 倾倒保护：roll 或 pitch > 0.8 rad 时自动进入阻尼模式")
    print("  • 超时保护：lowstate 超过 0.2s 未更新则自动退出")
    print("=" * 70 + "\n")

    controller = G1Sim2RealController(cfg, args.model, iface=args.iface, dry_run=args.dry_run)
    controller.run(policy_registry)


if __name__ == "__main__":
    main()
