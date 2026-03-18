#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Go1 Sim2Sim (MuJoCo) and Sim2Real (Unitree SDK) controller

Usage:
    Sim:  python sim2real_walk.py --mode sim  [--model policy.pt] [--config go1_walk.yaml]
    Real: python sim2real_walk.py --mode real [--model policy.pt] [--config go1_walk.yaml]
"""

import time
import numpy as np
import argparse
import torch
import sys
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from joystick import create_gamepad_controller

# ========== Config loader ==========

def load_config(config_path):
    script_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    cfg = SimpleNamespace(**raw)
    for k, v in raw.items():
        if isinstance(v, list):
            if k in ('joint_names_mujoco', 'actuator_names_mujoco', 'sdk_joint_order'):
                setattr(cfg, k, v)
                continue
            try:
                v = np.array(v, dtype=np.int32 if 'map' in k else np.float32)
            except (ValueError, TypeError):
                pass
        setattr(cfg, k, v)

    # resolve relative paths
    for attr in ('policy_path', 'xml_path'):
        val = getattr(cfg, attr, None)
        if val and (val.startswith('./') or val.startswith('../')):
            setattr(cfg, attr, os.path.abspath(os.path.join(script_dir, val)))

    if hasattr(cfg, 'vx_range'):   cfg.vx_range   = tuple(cfg.vx_range)
    if hasattr(cfg, 'vy_range'):   cfg.vy_range   = tuple(cfg.vy_range)
    if hasattr(cfg, 'vyaw_range'): cfg.vyaw_range = tuple(cfg.vyaw_range)
    return cfg


# ========== Shared helpers ==========

def quat_rotate_inverse(q, v):
    q_w, q_vec = q[3], q[:3]
    return v * (2.0 * q_w**2 - 1.0) - np.cross(q_vec, v) * q_w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0


def build_obs(base_ang_vel, proj_grav, commands, dof_pos_rel, dof_vel, last_action, cfg):
    """45-dim observation (single frame, no history for Go1 baseline policy)."""
    obs = np.concatenate([
        base_ang_vel  * cfg.obs_scales['base_ang_vel'],
        proj_grav,
        commands,
        dof_pos_rel   * cfg.obs_scales['joint_pos'],
        dof_vel       * cfg.obs_scales['joint_vel'],
        last_action,
    ]).astype(np.float32)
    return obs


def infer(policy, obs_np):
    with torch.no_grad():
        out = policy(torch.from_numpy(obs_np[np.newaxis]))
        if isinstance(out, tuple): out = out[0]
    return out.cpu().numpy().flatten().astype(np.float32)


# ========== Sim2Sim (MuJoCo) ==========

class Sim2SimController:
    def __init__(self, cfg, model_name):
        import mujoco, mujoco.viewer
        from scipy.spatial.transform import Rotation as R
        self._mujoco = mujoco
        self._viewer_mod = mujoco.viewer
        self._R = R
        self.cfg = cfg

        base_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path    = os.path.join(base_dir, 'assets', os.path.basename(cfg.xml_path))
        policy_path = os.path.join(base_dir, 'exported_policy', model_name)

        print(f"Loading MuJoCo model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg.sim_dt
        self.policy_decimation  = int(cfg.policy_dt / cfg.sim_dt)

        self.joint_qpos_addrs = [self.model.jnt_qposadr[self.model.joint(n).id] for n in cfg.joint_names_mujoco]
        self.joint_qvel_addrs = [self.model.jnt_dofadr [self.model.joint(n).id] for n in cfg.joint_names_mujoco]
        self.actuator_ids     = [self.model.actuator(n).id                       for n in cfg.actuator_names_mujoco]
        self.num_joints       = len(cfg.joint_names_mujoco)

        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location='cpu')

        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.kp = cfg.kp_walk
        self.kd = cfg.kd_walk

        # default pose in MuJoCo order
        self.default_qpos_mj = cfg.default_qpos_isaac[cfg.isaac_to_mujoco_map]
        self.target_qpos_mj  = self.default_qpos_mj.copy()

        self.data.qpos[self.joint_qpos_addrs] = self.default_qpos_mj
        self.data.qpos[2] = getattr(cfg, 'init_height', 0.35)
        mujoco.mj_step(self.model, self.data)
        print("Sim2Sim controller initialized")

    def _pd(self):
        tau = self.kp * (self.target_qpos_mj - self.data.qpos[self.joint_qpos_addrs]) \
            + self.kd * (0.0              - self.data.qvel[self.joint_qvel_addrs])
        self.data.ctrl[self.actuator_ids] = tau

    def run(self, gamepad):
        mujoco, viewer_mod, R = self._mujoco, self._viewer_mod, self._R
        cfg = self.cfg
        motiontime = 0
        start_time = None

        with viewer_mod.launch_passive(self.model, self.data) as viewer:
            viewer.cam.lookat[:] = self.data.qpos[:3]
            viewer.cam.distance  = 2.0
            viewer.cam.azimuth   = 90
            viewer.cam.elevation = -20
            start_time = time.time()

            while viewer.is_running():
                if gamepad.exit_requested:
                    break

                self._pd()
                mujoco.mj_step(self.model, self.data)
                motiontime += 1

                if motiontime % self.policy_decimation == 0:
                    quat_wxyz = self.data.qpos[3:7]
                    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
                    proj_grav = R.from_quat(quat_xyzw).apply([0, 0, -1], inverse=True)
                    base_ang  = self.data.qvel[3:6]

                    curr_qpos_mj = self.data.qpos[self.joint_qpos_addrs]
                    curr_qvel_mj = self.data.qvel[self.joint_qvel_addrs]
                    dof_pos_rel  = curr_qpos_mj[cfg.mujoco_to_isaac_map] - cfg.default_qpos_isaac
                    dof_vel      = curr_qvel_mj[cfg.mujoco_to_isaac_map]

                    cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                    commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)

                    obs    = build_obs(base_ang, proj_grav, commands, dof_pos_rel, dof_vel, self.last_action, cfg)
                    action = infer(self.policy, obs)

                    self.last_action    = action.copy()
                    target_isaac        = action * cfg.action_scale['pos'] + cfg.default_qpos_isaac
                    self.target_qpos_mj = target_isaac[cfg.isaac_to_mujoco_map]

                viewer.cam.lookat[:] = self.data.qpos[:3]
                viewer.sync()

                expected = start_time + motiontime * cfg.sim_dt
                sleep    = expected - time.time()
                if sleep > 0:
                    time.sleep(sleep)

                if motiontime % int(1.0 / cfg.sim_dt) == 0:
                    rt = time.time() - start_time
                    vx, vy, vyaw = gamepad.get_velocity()
                    print(f"[Gamepad] vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}")
                    print(f"[Sim] t={motiontime*cfg.sim_dt:.1f}s h={self.data.qpos[2]:.3f}m "
                          f"[Real] t={rt:.1f}s hz={motiontime/rt:.1f}")


# ========== Sim2Real (Unitree SDK) ==========

class Sim2RealController:
    """Go1 real robot controller via unitree_legged_sdk."""

    # SDK motor index: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
    LOWLEVEL = 0xff

    def __init__(self, cfg, model_name):
        self.cfg = cfg

        # --- Load SDK ---
        sdk_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'unitree_legged_sdk', 'lib', 'python', 'amd64'))
        if sdk_dir not in sys.path:
            sys.path.insert(0, sdk_dir)
        import robot_interface as sdk
        self.sdk = sdk

        self.udp       = sdk.UDP(self.LOWLEVEL, cfg.local_port, cfg.robot_ip, cfg.robot_port)
        self.safe      = sdk.Safety(sdk.LeggedType.Go1)
        self.low_cmd   = sdk.LowCmd()
        self.low_state = sdk.LowState()
        self.udp.InitCmdData(self.low_cmd)
        print(f"UDP: {cfg.robot_ip}:{cfg.robot_port}")

        # --- Load policy ---
        base_dir    = os.path.dirname(os.path.abspath(__file__))
        policy_path = os.path.join(base_dir, 'exported_policy', model_name)
        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location='cpu')

        self.last_action = np.zeros(12, dtype=np.float32)
        # target in SDK order, held between policy steps
        self.target_sdk  = cfg.default_qpos_isaac[cfg.isaac_to_sdk_map].copy()
        self.policy_decimation = int(cfg.policy_dt / cfg.sim_dt)

    # ------------------------------------------------------------------

    def _recv(self):
        self.udp.Recv()
        self.udp.GetRecv(self.low_state)

    def _send(self, target_sdk, kp, kd):
        for i in range(12):
            self.low_cmd.motorCmd[i].q   = float(target_sdk[i])
            self.low_cmd.motorCmd[i].dq  = 0.0
            self.low_cmd.motorCmd[i].Kp  = float(kp[i]) if hasattr(kp, '__len__') else float(kp)
            self.low_cmd.motorCmd[i].Kd  = float(kd[i]) if hasattr(kd, '__len__') else float(kd)
            self.low_cmd.motorCmd[i].tau = 0.0
        self.safe.PowerProtect(self.low_cmd, self.low_state, 1)
        self.udp.SetSend(self.low_cmd)
        self.udp.Send()

    def _get_obs_parts(self):
        """Extract (base_ang_vel, proj_grav, dof_pos_isaac, dof_vel_isaac) from low_state."""
        from scipy.spatial.transform import Rotation as R

        gyro = np.array(self.low_state.imu.gyroscope, dtype=np.float32)

        rpy  = np.array(self.low_state.imu.rpy, dtype=np.float32)
        quat_xyzw = R.from_euler('xyz', rpy).as_quat()
        proj_grav = quat_rotate_inverse(quat_xyzw, np.array([0, 0, -1], dtype=np.float32))

        q_sdk  = np.array([self.low_state.motorState[i].q  for i in range(12)], dtype=np.float32)
        dq_sdk = np.array([self.low_state.motorState[i].dq for i in range(12)], dtype=np.float32)

        dof_pos = q_sdk [self.cfg.sdk_to_isaac_map]
        dof_vel = dq_sdk[self.cfg.sdk_to_isaac_map]
        return gyro, proj_grav, dof_pos, dof_vel

    # ------------------------------------------------------------------

    def _wait_for_connection(self):
        print("Connecting to robot (damping mode)...")
        for i in range(12):
            self.low_cmd.motorCmd[i].q   = 0.0
            self.low_cmd.motorCmd[i].dq  = 0.0
            self.low_cmd.motorCmd[i].Kp  = 0.0
            self.low_cmd.motorCmd[i].Kd  = 3.0
            self.low_cmd.motorCmd[i].tau = 0.0
        for _ in range(100):
            self._recv()
            self.udp.SetSend(self.low_cmd)
            self.udp.Send()
            time.sleep(self.cfg.sim_dt)

        q_sum = sum(abs(self.low_state.motorState[i].q) for i in range(12))
        if q_sum < 0.01:
            print("ERROR: No joint data received. Check power and network.")
            return False
        print("Robot connected.")
        return True

    def run(self, gamepad):
        if not self._wait_for_connection():
            return

        cfg = self.cfg
        motiontime   = 0
        policy_count = 0
        start_time   = time.time()

        # initial joint positions (SDK order) for standup interpolation
        self._recv()
        q0_sdk = np.array([self.low_state.motorState[i].q for i in range(12)], dtype=np.float32)
        default_sdk = cfg.default_qpos_isaac[cfg.isaac_to_sdk_map]

        print("Starting control loop (200 Hz)...")
        while True:
            loop_start = time.time()
            self._recv()

            # update gamepad from wireless_remote (UnitreeSDK gamepad)
            gamepad.update(list(self.low_state.wirelessRemote))

            if gamepad.exit_requested:
                print("Exit requested.")
                break

            sim_time = motiontime * cfg.sim_dt

            # --- Phase 1: stand up ---
            if sim_time < cfg.standup_duration:
                rate = sim_time / cfg.standup_duration
                target_sdk = q0_sdk * (1.0 - rate) + default_sdk * rate
                kp = np.full(12, 20.0)
                kd = np.full(12, 0.5)

            # --- Phase 2: stabilize ---
            elif sim_time < cfg.standup_duration + cfg.stabilize_duration:
                target_sdk = default_sdk.copy()
                kp = cfg.kp_walk
                kd = cfg.kd_walk

            # --- Phase 3: policy ---
            else:
                # tilt check
                rpy = np.array(self.low_state.imu.rpy, dtype=np.float32)
                if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                    print(f"\nWARNING: tilt roll={rpy[0]:.2f} pitch={rpy[1]:.2f}, stopping.")
                    break

                policy_count += 1
                if policy_count >= self.policy_decimation:
                    policy_count = 0

                    base_ang, proj_grav, dof_pos, dof_vel = self._get_obs_parts()
                    dof_pos_rel = dof_pos - cfg.default_qpos_isaac

                    cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                    commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)

                    obs    = build_obs(base_ang, proj_grav, commands, dof_pos_rel, dof_vel, self.last_action, cfg)
                    action = infer(self.policy, obs)

                    self.last_action = action.copy()
                    target_isaac     = action * cfg.action_scale['pos'] + cfg.default_qpos_isaac
                    self.target_sdk  = target_isaac[cfg.isaac_to_sdk_map]

                target_sdk = self.target_sdk
                kp = cfg.kp_walk
                kd = cfg.kd_walk

            self._send(target_sdk, kp, kd)
            motiontime += 1

            # status print ~1Hz
            if motiontime % int(1.0 / cfg.sim_dt) == 0:
                vx, vy, vyaw = gamepad.get_velocity()
                rpy = np.array(self.low_state.imu.rpy, dtype=np.float32)
                print(f"[t={sim_time:.1f}s] vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f} "
                      f"roll={rpy[0]:.2f} pitch={rpy[1]:.2f}")

            elapsed = time.time() - loop_start
            sleep   = cfg.sim_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

        # safe stop: damping
        for i in range(12):
            self.low_cmd.motorCmd[i].Kp = 0.0
            self.low_cmd.motorCmd[i].Kd = 2.0
        self.udp.SetSend(self.low_cmd)
        self.udp.Send()
        print("Control stopped.")


# ========== Main ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   type=str, default='sim',          help='sim | real')
    parser.add_argument('--model',  type=str, default='policy.pt')
    parser.add_argument('--config', type=str, default='go1_walk.yaml')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve(name, subdir):
        if os.path.exists(name): return name
        return os.path.join(script_dir, subdir, name)

    cfg = load_config(resolve(args.config, 'config'))

    if args.mode == 'real':
        gamepad_type = getattr(cfg, 'gamepad_type_sim2real', 'unitree_sdk')
    else:
        gamepad_type = getattr(cfg, 'gamepad_type_sim2sim', 'unitree_pygame')

    gamepad = create_gamepad_controller(
        gamepad_type,
        vx_range=cfg.vx_range,
        vy_range=cfg.vy_range,
        vyaw_range=cfg.vyaw_range,
    )
    gamepad.start()

    print("\n" + "="*60)
    print(f"Go1 {'Sim2Real' if args.mode == 'real' else 'Sim2Sim'} | gamepad={gamepad_type}")
    print("  Left stick  : vx (fwd/back) / vy (strafe)")
    print("  Right stick : vyaw (turn)")
    print("  RB+A        : Walk policy")
    print("  Start       : Exit")
    print("="*60 + "\n")

    if args.mode == 'sim':
        controller = Sim2SimController(cfg, args.model)
        controller.run(gamepad)
    else:
        controller = Sim2RealController(cfg, args.model)
        controller.run(gamepad)

    gamepad.stop()
    print("Done.")
