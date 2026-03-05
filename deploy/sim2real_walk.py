#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Unified Sim2Sim (MuJoCo) and Sim2Real (Unitree SDK) controller

Supports two modes:
1. Sim2Sim: MuJoCo simulation (--mode sim)
2. Sim2Real: Unitree Go1 real robot (--mode real)

Shared configuration parameters and policy inference logic.

Usage:
    Simulation: python sim2sim_sim2real_unified.py --mode sim
    Real robot: python sim2sim_sim2real_unified.py --mode real
"""

import time
import numpy as np
import argparse
import torch
import threading
import sys
import os
from joystick import RemoteController, apply_deadzone
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import yaml
from legged_gym import LEGGED_GYM_ROOT_DIR

# ========== Configuration loader ==========
def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 获取当前脚本所在目录，用于处理 YAML 内部的相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    class Config:
        def __init__(self, cfg):
            for key, value in cfg.items():
                # --- 新增：路径处理逻辑 ---
                if isinstance(value, str) and ('/' in value or '\\' in value):
                    # 替换环境变量占位符
                    if "{LEGGED_GYM_ROOT_DIR}" in value:
                        # 假设你已经定义了该环境变量，或者手动指定
                        value = value.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
                    
                    # 如果是相对路径，将其转为基于脚本位置的绝对路径
                    if value.startswith("./") or value.startswith("../"):
                        value = os.path.abspath(os.path.join(script_dir, value))
                
                # 标准转换逻辑
                if isinstance(value, dict):
                    setattr(self, key, value)
                elif isinstance(value, list):
                    if key in ('sdk_joint_order', 'leg_order', 'joint_suffixes'):
                        setattr(self, key, value)
                    else:
                        try:
                            setattr(self, key, np.array(value, dtype=np.float32))
                        except:
                            setattr(self, key, value)
                else:
                    setattr(self, key, value)
            
            # 自动计算 policy_dt 等 (保留原有逻辑)
            if hasattr(self, 'policy_hz'): self.policy_dt = 1.0 / self.policy_hz
            if hasattr(self, 'vx_range'): self.vx_range = tuple(self.vx_range)
            if hasattr(self, 'vy_range'): self.vy_range = tuple(self.vy_range)
            if hasattr(self, 'vyaw_range'): self.vyaw_range = tuple(self.vyaw_range)

    return Config(config_dict)

# ========== Helper functions ==========

def quat_rotate_inverse(q, v):
    """Rotate a vector from world frame to body frame using the inverse quaternion."""
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def compute_projected_gravity(quat):
    """Compute the projected gravity vector in the body frame."""
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    projected_gravity = quat_rotate_inverse(quat, gravity_world)
    return projected_gravity

def build_obs(base_ang_vel, projected_gravity, commands, dof_pos, dof_vel, last_action, config):
    """
    Build the observation vector (45 dims):
    1-3:   base angular velocity [wx, wy, wz] scaled by ang_vel
    4-6:   projected gravity [gx, gy, gz]
    7-9:   commands [lin_vel_x, lin_vel_y, ang_vel_yaw] scaled by commands
    10-21: joint position delta (dof_pos - default_dof_pos) scaled by dof_pos
    22-33: joint velocities scaled by dof_vel
    34-45: last actions
    """
    obs = []
    
    # 1-3: Base angular velocity (scaled)
    obs.extend(list(base_ang_vel * config.obs_scales['ang_vel']))
    
    # 4-6: Projected gravity
    obs.extend(list(projected_gravity))
    
    # 7-9: Commands (scaled)
    commands_scaled = commands * config.obs_scales['commands']
    obs.extend(list(commands_scaled))
    
    # 10-21: dof_pos - default_dof_pos (scaled)
    pos_delta = (dof_pos - config.default_dof_pos) * config.obs_scales['dof_pos']
    obs.extend(list(pos_delta))
    
    # 22-33: dof_vel (scaled)
    obs.extend(list(dof_vel * config.obs_scales['dof_vel']))
    
    # 34-45: Last action
    obs.extend(list(last_action))
    
    return np.array(obs, dtype=np.float32)


# ========== Gamepad controller ==========

class GamepadController:
    """Thread-safe gamepad controller (Logitech F710 - Linux native interface)."""
    def __init__(self, vx_range=(0.0, 1.2), vy_range=(-0.3, 0.3), vyaw_range=(-1.57, 1.57)):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vyaw_range = vyaw_range
        self.lock = threading.Lock()
        self.running = True
        self.exit_requested = False
        self.thread = None
        
        # Initialize gamepad (Linux native device)
        try:
            self.gamepad = RemoteController()
            self.gamepad.start()
            print("✅ Gamepad initialized successfully (Linux native)")
        except Exception as e:
            print(f"❌ Failed to initialize gamepad: {e}")
            self.gamepad = None
        
        # Deadzone (normalized)
        self.deadzone = 0.05  # 5% deadzone
        
        # Velocity smoothing parameters (exponential moving average)
        self.alpha = 0.6  # smoothing: 60% new + 40% old (faster response)
        self.vx_smooth = 0.0
        self.vy_smooth = 0.0
        self.vyaw_smooth = 0.0
        
        # Speed step control (D-pad incremental adjustment)
        self.vx_increment = 0.1  # change per press: 0.1 m/s
        self.vx_target = 0.0     # target speed step
        self.dpad_last_state = {'up': False, 'down': False}  # edge detection
    
    def get_velocity(self):
        with self.lock:
            return self.vx, self.vy, self.vyaw
    
    def set_velocity(self, vx, vy, vyaw):
        with self.lock:
            self.vx = np.clip(vx, self.vx_range[0], self.vx_range[1])
            self.vy = np.clip(vy, self.vy_range[0], self.vy_range[1])
            self.vyaw = np.clip(vyaw, self.vyaw_range[0], self.vyaw_range[1])
    
    def gamepad_thread(self):
        """Gamepad reading thread - synchronized with policy frequency."""
        if self.gamepad is None:
            print("Gamepad not available, using zero velocity")
            return
        
        # Sync with policy frequency: 33Hz
        update_interval = 1.0 / 33.0  # 0.0303s
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Read stick values (normalized to [-1, 1])
                left_x, left_y = self.gamepad.get_left_stick(normalize=True)
                right_x, right_y = self.gamepad.get_right_stick(normalize=True)
                
                # Apply deadzone
                left_x = apply_deadzone(left_x, self.deadzone)
                left_y = apply_deadzone(left_y, self.deadzone)
                right_x = apply_deadzone(right_x, self.deadzone)
                
                # D-pad incremental control (HAT axes: 6=X, 7=Y)
                with self.gamepad.lock:
                    dpad_y = self.gamepad.axes[7] if len(self.gamepad.axes) > 7 else 0  # Y axis: -32767=up, +32767=down
                
                dpad_up_pressed = (dpad_y < -16000)    # up
                dpad_down_pressed = (dpad_y > 16000)   # down
                
                # Edge detection: trigger on press (not hold)
                if dpad_up_pressed and not self.dpad_last_state['up']:
                    self.vx = min(self.vx + self.vx_increment, self.vx_range[1])
                    print(f"\n[D-pad UP] speed step: {self.vx:.1f} m/s")
                
                if dpad_down_pressed and not self.dpad_last_state['down']:
                    self.vx = max(self.vx - self.vx_increment, 0.0)  # min 0, no backward
                    print(f"\n[D-pad DOWN] speed step: {self.vx:.1f} m/s")
                
                # Update D-pad state
                self.dpad_last_state['up'] = dpad_up_pressed
                self.dpad_last_state['down'] = dpad_down_pressed
                
                # Map sticks / D-pad to velocities
                # Priority: D-pad speed step; sticks act as fine control
                # Left stick Y: -1 (push up) to +1 (push down)
                if abs(left_y) > 0.1:  # use stick if significant input
                    if left_y <= 0:  # push up (negative Y)
                        self.vx = (-left_y) * self.vx_range[1]  # map to [0, max]
                    else:  # push down
                        self.vx = 0.0  # backward not supported
                else:  # stick centered, keep D-pad speed
                    self.vx = self.vx

                # Left stick X: lateral velocity mapping
                self.vy = -left_x * (self.vy_range[1])   # -0.3 .. +0.3 m/s

                # Right stick X: yaw rate mapping
                self.vyaw = -right_x * self.vyaw_range[1]
                
                
                # Update (clamped) velocity values
                self.set_velocity(self.vx, self.vy, self.vyaw)
                
                # Check exit button (Start = button 7)
                if self.gamepad.is_button_pressed(self.gamepad.BTN_START):
                    print("\n✅ Start button pressed - exiting")
                    self.exit_requested = True
                    break
                
                # (Periodic Gamepad print removed; status printed from simulation loop)
                
                # Sync with policy frequency: 33Hz (update every 0.03s)
                elapsed = time.time() - loop_start
                sleep_time = max(0, update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"\nGamepad error: {e}")
                time.sleep(0.1)
    
    def start(self):
        self.thread = threading.Thread(target=self.gamepad_thread, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.gamepad:
            self.gamepad.stop()
        if self.thread:
            self.thread.join(timeout=1.0)


# ========== Sim2Sim (MuJoCo) controller ==========

class Sim2SimController:
    """MuJoCo simulation controller."""
    
    def __init__(self, config, model_name):
        self.config = config
        
        xml_path = config.xml_path
        policy_path = os.path.join(config.policy_path, model_name)
        
        # Generate joint and actuator names from configuration
        self.joint_names = []
        self.actuator_names = []
        for leg in config.leg_order:
            for suffix in config.joint_suffixes:
                self.joint_names.append(f"{leg}_{suffix}_joint")
                self.actuator_names.append(f"{leg}_{suffix}")
        
        # Load MuJoCo
        self.mujoco = mujoco
        self.mujoco_viewer = mujoco.viewer
        
        print(f"Loading MuJoCo model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.sim_dt
        
        # Get joint qpos/dof addresses and actuator ids
        self.joint_qpos_addrs = []
        self.joint_dof_addrs = []
        self.actuator_ids = []
        
        for joint_name, actuator_name in zip(self.joint_names, self.actuator_names):
            joint_id = self.model.joint(joint_name).id
            qpos_addr = self.model.jnt_qposadr[joint_id]
            dof_addr = self.model.jnt_dofadr[joint_id]
            actuator_id = self.model.actuator(actuator_name).id
            
            self.joint_qpos_addrs.append(qpos_addr)
            self.joint_dof_addrs.append(dof_addr)
            self.actuator_ids.append(actuator_id)
        
        # Load policy
        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location='cpu')
        self.policy.eval()
        
        # Initialize state
        self.last_action = np.zeros(12, dtype=np.float32)
        self.qDes = np.zeros(12, dtype=np.float32)
        
        # Policy frequency control
        self.policy_decimation = int(config.policy_dt / config.sim_dt)
        self.policy_counter = 0
        
        # Initialize robot pose
        for i, qpos_addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[qpos_addr] = config.default_dof_pos[i]
        self.data.qpos[2] = 0.35  # initial height
        mujoco.mj_forward(self.model, self.data)
        
        print(f"Sim2Sim controller initialized")
        
    
    def get_state(self):
        """Get current robot state needed for observation."""
        # Base angular velocity
        base_ang_vel = self.data.qvel[3:6].copy()
        
        # Projected gravity
        quat_wxyz = self.data.qpos[3:7].copy()
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        projected_gravity = compute_projected_gravity(quat_xyzw)
        
        # Joint positions and velocities
        dof_pos = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addrs], dtype=np.float32)
        dof_vel = np.array([self.data.qvel[addr] for addr in self.joint_dof_addrs], dtype=np.float32)
        
        return base_ang_vel, projected_gravity, dof_pos, dof_vel
    
    def send_command(self, target_pos):
        """Send control commands to MuJoCo actuators."""
        for i, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = target_pos[i]
    
    def step(self):
        """Advance the MuJoCo simulation by one timestep."""
        self.mujoco.mj_step(self.model, self.data)
    
    def run(self, gamepad):
        """
        Main simulation loop with Absolute Time Sync and Render Decimation.
        """
        motiontime = 0 # simulation step counter 
        
        # --- Time Synchronization Setup ---
        sim_dt = self.model.opt.timestep  # Physics timestep (e.g., 0.005s)
        target_render_fps = 50            # Human eye only needs 30-60 FPS
        # Calculate how many physics steps to skip between renders
        render_skip = int(1.0 / (target_render_fps * sim_dt))
        if render_skip < 1: render_skip = 1
        
        # Launch viewer
        with self.mujoco_viewer.launch_passive(self.model, self.data) as viewer:
            # Initial Camera setup
            viewer.cam.lookat[:] = self.data.qpos[:3]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            
            # --- Establish Absolute Time Reference ---
            # Record start time immediately before entering the loop
            start_time = time.time() 
            
            while viewer.is_running():
                if gamepad.exit_requested:
                    print("\nExit request detected, ending starget_posimulation...")
                    break
                
                # Get MuJoCo internal simulation time
                sim_time = self.data.time
                
                # --- Phase 1: Stand up (Linear Interpolation) ---
                if sim_time <= self.config.standup_duration:
                    rate = min(sim_time / self.config.standup_duration, 1.0)
                    for i, qpos_addr in enumerate(self.joint_qpos_addrs):
                        current_q = self.data.qpos[qpos_addr]
                        self.qDes[i] = current_q * (1 - rate) + self.config.default_dof_pos[i] * rate
                    self.send_command(self.qDes)
                
                # --- Phase 2: Stabilize at Default Pose ---
                elif sim_time <= self.config.standup_duration + self.config.stabilize_duration:
                    self.qDes = self.config.default_dof_pos.copy()
                    self.send_command(self.qDes)
                
                # --- Phase 3: Neural Network Policy Control ---
                else:
                    # Safety check: Detect if robot is falling
                    quat_wxyz = self.data.qpos[3:7].copy()
                    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                    rpy = R.from_quat(quat_xyzw).as_euler('xyz')
                    if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                        print(f"\nWarning at {sim_time:.2f}s: Robot tilted! roll={rpy[0]:.2f}, pitch={rpy[1]:.2f}")
                    
                    # Policy inference (decimated frequency)
                    self.policy_counter += 1
                    if self.policy_counter >= self.policy_decimation:
                        self.policy_counter = 0
                        
                        # Get user input from gamepad
                        cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                        commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                        
                        # Extract robot state
                        base_ang_vel, projected_gravity, dof_pos, dof_vel = self.get_state()
                        
                        # Prepare observation for the policy
                        obs = build_obs(base_ang_vel, projected_gravity, commands, 
                                        dof_pos, dof_vel, self.last_action, self.config)
                        obs_batch = obs[np.newaxis, :].astype(np.float32)
                        
                        # Forward pass through the neural network
                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(obs_batch)
                            action_tensor = self.policy(obs_tensor)
                            if isinstance(action_tensor, tuple):
                                action_tensor = action_tensor[0]
                            action = action_tensor.cpu().numpy().flatten().astype(np.float32)
                        
                        # Scale action to joint targets
                        self.last_action = action[:12].copy()
                        self.qDes = action[:12] * self.config.action_scale + self.config.default_dof_pos
        
                    self.send_command(self.qDes)
                
                # --- Physics Step ---
                self.step()
                motiontime += 1 # 注意 这个要放在整个while循环的最后面，不能只放在policy控制的部分，因为standup和stabilize阶段也需要计数
                
                # --- Visual Update (Render Decimation) ---
                # Syncing every step is slow; sync at 50Hz for better performance
                if motiontime % render_skip == 0:
                    viewer.cam.lookat[:] = self.data.qpos[:3]
                    viewer.sync()
                
                # --- Soft Real-time Synchronization (Absolute) ---
                # Calculate the exact time we SHOULD be at
                expected_real_time = start_time + (motiontime * sim_dt)
                time_to_sleep = expected_real_time - time.time()
                
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                
                # --- Status Telemetry (Original Format) ---
                if motiontime % int(1.0 / self.config.sim_dt) == 0:
                    real_time_now = time.time() - start_time
                    actual_hz = motiontime / real_time_now if real_time_now > 0 else 0
                    
                    vx_cur, vy_cur, vyaw_cur = gamepad.get_velocity()
                    
                    print(f"[Gamepad] vx={vx_cur:+.2f} m/s | vy={vy_cur:+.2f} | yaw={vyaw_cur:+.2f} rad/s")
                    print(f"[Sim Time]: t={self.data.time:.1f}s, Base height: {self.data.qpos[2]:.3f}m")
                    print(f"[Real Time]: t={real_time_now:.1f}s, Actual Hz: {actual_hz:.2f} Hz")


# ========== Sim2Real (Unitree SDK) controller ==========

class Sim2RealController:
    """Unitree Go1 real robot controller."""
    
    def __init__(self, config, policy_path):
        self.config = config
        
        # Import Unitree SDK
        SDK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'unitree_legged_sdk', 'lib', 'python', 'amd64'))
        sys.path.append(SDK_DIR)
        import robot_interface as sdk
        self.sdk = sdk
        
        # SDK Index
        self.index = config.index
        self.order = config.joint_order
        
        # Initial UDP
        LOWLEVEL = 0xff
        self.udp = sdk.UDP(LOWLEVEL, config.local_port, 
                          config.robot_ip, config.robot_port)
        self.low_cmd = sdk.LowCmd()
        self.low_state = sdk.LowState()
        self.udp.InitCmdData(self.low_cmd)
        
        print(f"UDP initialized: {config.robot_ip}:{config.robot_port}")
        
        # Load policy
        self.policy = torch.jit.load(policy_path, map_location='cpu')
        self.policy.eval()
        print(f"Loading policy: {policy_path}")
        
        # Initialize state
        self.last_action = np.zeros(12, dtype=np.float32)
        self.qDes_train = np.zeros(12, dtype=np.float32)
        
        # Policy frequency control
        self.policy_decimation = int(config.policy_dt / config.sim_dt)
        self.policy_counter = 0

        # Cache the latest SDK command for constant 200Hz sending
        self.current_qDes_sdk = None
        self.current_kp = config.kp_walk
        self.current_kd = config.kd_walk
        
        print("Sim2Real controller initialized")
    
    def wait_for_connection(self):
        """Wait for robot connection and ensure data flow."""
        print("Waiting for robot connection...")

        # Initialize command (damping mode, Kp=0)
        for i in range(12):
            self.low_cmd.motorCmd[i].q = 0.0
            self.low_cmd.motorCmd[i].dq = 0.0
            self.low_cmd.motorCmd[i].Kp = 0.0
            self.low_cmd.motorCmd[i].Kd = 3.0
            self.low_cmd.motorCmd[i].tau = 0.0

        # Send commands to activate communications
        for i in range(100):
            self.udp.Recv()
            self.udp.GetRecv(self.low_state)
            self.udp.SetSend(self.low_cmd)
            self.udp.Send()
            time.sleep(self.config.sim_dt)

        # Check if valid joint data has been received
        q_sum = sum(abs(self.low_state.motorState[i].q) for i in range(12))
        if q_sum < 0.01:
            print("Error: No valid joint data received!")
            print("Please check:")
            print("  1. Robot is powered on")
            print("  2. Network connection is working")
            print("  3. IP address is correct (current: {})".format(self.config.robot_ip))
            return False

        print("Robot connected successfully!")
        self._print_state()
        return True
    
    def get_state(self):
        """Get the current state (format conversion only, no UDP receive).

        Note: ensure `run()` has updated `self.low_state` before calling.
        """
        # Base angular velocity (SDK format)
        base_ang_vel = np.array([
            self.low_state.imu.gyroscope[0],
            self.low_state.imu.gyroscope[1],
            self.low_state.imu.gyroscope[2]
        ], dtype=np.float32)
        
        # Projected gravity from IMU
        rpy = np.array(self.low_state.imu.rpy, dtype=np.float32)
        quat = R.from_euler('xyz', [rpy[0], rpy[1], rpy[2]]).as_quat()
        projected_gravity = compute_projected_gravity(quat)
        
        # Joint positions and velocities (SDK -> training order)
        q_sdk = np.array([self.low_state.motorState[i].q for i in range(12)], dtype=np.float32)
        dq_sdk = np.array([self.low_state.motorState[i].dq for i in range(12)], dtype=np.float32)
        
        dof_pos = q_sdk[self.config.sdk_to_train_map]
        dof_vel = dq_sdk[self.config.sdk_to_train_map]
        
        return base_ang_vel, projected_gravity, dof_pos, dof_vel
    
    def send_command(self, target_sdk, kp, kd):
        """Send motor commands via SDK.

        Args:
            target_sdk: target joint angles in SDK order (12-d array)
            kp: proportional PD gain
            kd: derivative PD gain
        """
        # Set motor commands
        for i, jname in enumerate(self.joint_order):
            self.low_cmd.motorCmd[self.index[jname]].q = float(target_sdk[i])
            self.low_cmd.motorCmd[self.index[jname]].dq = 0.0
            self.low_cmd.motorCmd[self.index[jname]].Kp = float(kp)
            self.low_cmd.motorCmd[self.index[jname]].Kd = float(kd)
            self.low_cmd.motorCmd[self.index[jname]].tau = 0.0
        

    
    def run(self, gamepad):
        """Run the real-robot control loop."""
        if not self.wait_for_connection():
            print("Failed to connect to robot!")
        
        while True:
            loop_start = time.time()  # record loop start time
            
            # ⚠️ First receive low_state
            self.udp.Recv()
            self.udp.GetRecv(self.low_state)
        
            # Run control step
            
            # Phase 1: stand up
            if sim_time <= self.config.standup_duration:
                rate = min(sim_time / self.config.standup_duration, 1.0)
                self.qDes_train = dof_pos * (1 - rate) + self.config.default_dof_pos * rate
                
                # Update cached command (SDK order)
                self.current_qDes_sdk = self.qDes_train[self.config.train_to_sdk_map]
                self.current_kp = self.config.kp_stand
                self.current_kd = self.config.kd_stand
            
            # Phase 2: Stabilize
            elif sim_time <= self.config.standup_duration + self.config.stabilize_duration:
                self.qDes_train = self.config.default_dof_pos.copy()
                
                # Update cached command (SDK order)
                self.current_qDes_sdk = self.qDes_train[self.config.train_to_sdk_map]
                self.current_kp = self.config.kp_walk
                self.current_kd = self.config.kd_walk
            
            # Phase 3: Policy control
            else:
                # Check tilt
                if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                    print("\n⚠️  WARNING: Robot tilted!")
                    print(f"roll={rpy[0]:.2f}, pitch={rpy[1]:.2f}")
                
                # Policy inference (33Hz: executed every 6 sim_dt)
                self.policy_counter += 1
                if self.policy_counter >= self.policy_decimation:
                    self.policy_counter = 0
                    
                    # Get commands from gamepad
                    cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                    commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                    
                    # Get state
                    base_ang_vel, projected_gravity, dof_pos, dof_vel = self.get_state()
                    
                    # Build observation
                    obs = build_obs(base_ang_vel, projected_gravity, commands,
                                    dof_pos, dof_vel, self.last_action, self.config)
                    obs_batch = obs[np.newaxis, :].astype(np.float32)
                    # Policy inference
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(obs_batch)
                        action_tensor = self.policy(obs_tensor)
                        if isinstance(action_tensor, tuple):
                            action_tensor = action_tensor[0]
                        action = action_tensor.cpu().numpy().flatten().astype(np.float32)
                    
                    # Scale action
                    self.last_action = action[:12].copy()
                    
                    self.qDes_train = action[:12] * self.config.action_scale + self.config.default_dof_pos
                    
                    # Update cached command (this will be sent repeatedly for the next frames)
                    self.current_qDes_sdk = self.qDes_train[self.config.train_to_sdk_map]
                    self.current_kp = self.config.kp_walk
                    self.current_kd = self.config.kd_walk
                    
        
            # Key fix: send commands every 200Hz loop regardless of inference
            if self.current_qDes_sdk is not None:
                
                self.send_command(self.current_qDes_sdk, self.current_kp, self.current_kd)
                # send commands (keeps UDP communication alive at 200Hz)
                self.udp.SetSend(self.low_cmd)
                self.udp.Send()
                time.sleep(self.config.control_dt)
            else:
                # Initialization phase: send damping command to avoid sudden motor motion
                damping_cmd = np.zeros(12, dtype=np.float32)
                self.send_command(damping_cmd, 0.0, 3.0)
                 
            # Precise timing: compensate for execution time
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.config.sim_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _control_step(self, sim_time, motiontime, gamepad):
        """Single control step (called at 200Hz)."""
        # Convert the previously received low_state into observation format
        
        rpy = np.array(self.low_state.imu.rpy, dtype=np.float32)
        
        

# ========== Main ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sim')
    parser.add_argument('--model', type=str, default='policy_45_continus.pt')
    parser.add_argument('--config', type=str, default='config/go1.yaml')
    args = parser.parse_args()
    
    # Adjust config path if it doesn't exist
    if not os.path.exists(args.config):
        args.config = os.path.join('config', args.config)
    
    # 1. 加载 Config (路径已经在内部处理好了)
    config = load_config(args.config)
    
    # 2. 初始化 Gamepad (使用 config 中的 range)
    gamepad = GamepadController(
        vx_range=config.vx_range,
        vy_range=config.vy_range,
        vyaw_range=config.vyaw_range
    )
    gamepad.start()
    
    print("\n" + "="*70)
    print("🎮 Gamepad Control (Logitech F710)")
    print("="*70)
    print("  Left Joystick:")
    print("    - Up/Down: Forward/Backward speed (vx)")
    print("    - Left/Right: Strafe speed (vy)")
    print("  Right Joystick:")
    print("    - Left/Right: Turn speed (vyaw)")
    print("  Start Button: Exit program")
    print("  Note: Release joystick to stop immediately")
    print("="*70 + "\n")
    
    # Create controller based on mode
    if args.mode == 'sim':
        # 3. 直接传入 config 和模型名称
        controller = Sim2SimController(config, args.model)
        controller.run(gamepad)
    else:
        # Sim2Real 同样逻辑
        policy_full_path = os.path.join(config.policy_path, args.model)
        controller = Sim2RealController(config, policy_full_path)
        controller.run(gamepad)
    
    # Stop gamepad controller
    gamepad.stop()
    print("\nProgram ended.")