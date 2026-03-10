#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Sim2Sim (MuJoCo)

Shared configuration parameters and policy inference logic.

Usage:
    Simulation: python deploy/sim2sim_walk.py --mode sim --model policy.pt
"""

import time
import numpy as np
import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from joystick import create_gamepad_controller
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import yaml
from pathlib import Path

LEGGED_RL_LAB_ROOT_DIR = str(Path(__file__).resolve().parent.parent)

# ========== Configuration loader ==========
def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 获取当前脚本所在目录，用于处理 YAML 内部的相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    class Config:
        def __init__(self, cfg):
            # 首先处理映射数组，确保它们是整数类型
            if 'isaac_to_mujoco_map' in cfg:
                self.isaac_to_mujoco_map = np.array(cfg['isaac_to_mujoco_map'], dtype=np.int32)
            if 'mujoco_to_isaac_map' in cfg:
                self.mujoco_to_isaac_map = np.array(cfg['mujoco_to_isaac_map'], dtype=np.int32)
            
            for key, value in cfg.items():
                # 跳过已经处理的映射数组
                if key in ('isaac_to_mujoco_map', 'mujoco_to_isaac_map'):
                    continue
                    
                # --- 新增：路径处理逻辑 ---
                if isinstance(value, str) and ('/' in value or '\\' in value):
                    # 替换环境变量占位符
                    if "{LEGGED_RL_LAB_ROOT_DIR}" in value:
                        value = value.replace("{LEGGED_RL_LAB_ROOT_DIR}", LEGGED_RL_LAB_ROOT_DIR)
                    
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

def build_obs(base_lin_vel, base_ang_vel, projected_gravity, commands, dof_pos_rel, dof_vel, last_action, config):
    """
    Build the observation vector matching IsaacLab velocity_env_cfg.py PolicyCfg:
    1-3:   base linear velocity (scaled by base_lin_vel)
    4-6:   base angular velocity (scaled by base_ang_vel)
    7-9:   projected gravity
    10-12: commands [vx, vy, vyaw]
    13-24: joint position relative to default (dof_pos_rel)
    25-36: joint velocities (scaled by joint_vel)
    37-48: last actions (12D position actions only)
    Total: 48 dims
    """
    obs = []
    
    # 1-3: Base linear velocity (scaled)
    base_lin_vel_scaled = base_lin_vel * config.obs_scales['base_lin_vel']
    obs.extend(list(base_lin_vel_scaled))
    
    # 4-6: Base angular velocity (scaled)
    base_ang_vel_scaled = base_ang_vel * config.obs_scales['base_ang_vel']
    obs.extend(list(base_ang_vel_scaled))
    
    # 7-9: Projected gravity
    obs.extend(list(projected_gravity))
    
    # 10-12: Commands [vx, vy, vyaw]
    obs.extend(list(commands))
    
    # 13-24: joint position relative to default
    obs.extend(list(dof_pos_rel))
    
    # 25-36: dof_vel (scaled)
    dof_vel_scaled = dof_vel * config.obs_scales['joint_vel']
    obs.extend(list(dof_vel_scaled))
    
    # 37-48: Last action
    obs.extend(list(last_action))
    
    return np.array(obs, dtype=np.float32)


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
        # joint_names (MuJoCo Order): FL_h,t,c | FR_h,t,c | RL_h,t,c | RR_h,t,c                           
                
        # Load MuJoCo
        self.mujoco = mujoco
        self.mujoco_viewer = mujoco.viewer
        
        print(f"Loading MuJoCo model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.sim_dt
        
        # MuJoCo Order: Get joint qpos/dof addresses and actuator ids
        # 注意 这里是将地址放入数组
        self.joint_qpos_addrs = [] #存储关节位置索引  0~6 是基座3d位置和4d四元数 7~18 是12个关节位置
        self.joint_dof_addrs = []  #存储关节速度索引  0~5 是基座3d线速度和3d角速度 6~17 是12个关节速度
        self.actuator_ids = []     #用于给电机发送控制指令 0~11 是12个电机，按照MuJoCo顺序
        
        for joint_name, actuator_name in zip(self.joint_names, self.actuator_names):
            #1 - 12
            joint_id = self.model.joint(joint_name).id
            #7 - 18
            qpos_addr = self.model.jnt_qposadr[joint_id]
            #6 - 17
            dof_addr = self.model.jnt_dofadr[joint_id]
            #0 - 11
            actuator_id = self.model.actuator(actuator_name).id

            self.joint_qpos_addrs.append(qpos_addr)
            self.joint_dof_addrs.append(dof_addr)
            self.actuator_ids.append(actuator_id)
        
        # Load policy
        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location='cpu')
        self.policy.eval()
        
        # Initialize state
        self.last_action = np.zeros(12, dtype=np.float32)  # 12D: position actions only
        self.qPos_isaac = np.zeros(12, dtype=np.float32)  # Target position (Isaac relative)
        self.qPos_mj = np.zeros(12, dtype=np.float32)      # Target position (MuJoCo)
        # self.qVel = np.zeros(12, dtype=np.float32)  # Target velocity (not used in position-only control)
        
        # Policy frequency control
        self.policy_decimation = config.control_decimation
        self.policy_counter = 0
        
        # Initialize robot pose
        self.default_dPos_mj = config.default_dPos_isaac[self.config.isaac_to_mujoco_map]
        # Set initial joint positions to default
        self.data.qpos[self.joint_qpos_addrs] = 0.0
        # Set initial joint velocities to zero
        self.data.qvel[self.joint_dof_addrs] = 0.0
        # Set initial base height
        self.data.qpos[2] = 0.40
        mujoco.mj_forward(self.model, self.data)
        
        print(f"Sim2Sim controller initialized")
        
    
    def get_state(self):
        """Get current robot state needed for observation."""
        # Get quaternion (MuJoCo is w,x,y,z)
        quat_wxyz = self.data.qpos[3:7] 
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        
        # Calculate Rotation Matrix
        r = R.from_quat(quat_xyzw)
        
        # Base Linear Velocity: World Frame -> Base Frame
        base_lin_vel = r.apply(self.data.qvel[0:3], inverse=True) 
        
        # Base Angular Velocity (MuJoCo freejoint qvel[3:6] is usually in local frame)
        base_ang_vel = self.data.qvel[3:6]
        
        projected_gravity = compute_projected_gravity(quat_xyzw)
        
        # Joint positions and velocities
        dof_pos_mujoco = self.data.qpos[self.joint_qpos_addrs]
        dof_vel_mujoco = self.data.qvel[self.joint_dof_addrs]
        
        # MuJoCo order -> Training order
        dof_pos_abs_train = dof_pos_mujoco[self.config.mujoco_to_isaac_map]
        dof_vel_train = dof_vel_mujoco[self.config.mujoco_to_isaac_map]
        dof_pos_rel = dof_pos_abs_train - self.config.default_dPos_isaac
        
        
        # Isaac order
        return base_lin_vel, base_ang_vel, projected_gravity, dof_pos_rel, dof_vel_train
    
    def send_command(self, target_pos_mj):
        """Send position command to MuJoCo actuators.
        """
        
        self.data.ctrl[self.actuator_ids] = target_pos_mj

            
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
        target_render_fps = 30            # Human eye only needs 30-60 FPS
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
                    print("\nExit request detected, ending simulation...")
                    break
                
                # Get MuJoCo internal simulation time
                sim_time = self.data.time
                
                base_lin, base_ang, proj_grav, curr_pos_rel, curr_vel = self.get_state()
                
 
                # --- Phase 1: Stand up (Linear Interpolation) ---
                if sim_time <= self.config.standup_duration:
                    # MuJoCo Order: Set to default position
                    self.qPos_mj = self.default_dPos_mj
                    self.qVel = np.zeros(12, dtype=np.float32)
                    print(f"dof_pos during standup: {curr_pos_rel + self.default_dPos_mj}") 
                    # Compute torques using PD control
                    self.send_command(self.qPos_mj)
                
                # --- Phase 2: Neural Network Policy Control ---
                else:
                    # Safety check: Detect if robot is falling
                    quat_wxyz = self.data.qpos[3:7].copy()
                    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                    rpy = R.from_quat(quat_xyzw).as_euler('xyz')

                    self.policy_counter += 1
                    if self.policy_counter == self.policy_decimation:
                        self.policy_counter = 0
                        
                        # Get user input from gamepad
                        # cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                        # commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                        
                        cmd_vx, cmd_vy, cmd_vyaw = [1.2, 0.0, 0.0]  # FOR TESTING ONLY
                        commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                        
                        # Prepare observation for the policy
                        obs = build_obs(base_lin, base_ang, proj_grav, commands, 
                                        curr_pos_rel, curr_vel, self.last_action, self.config)
                        obs_batch = obs[np.newaxis, :].astype(np.float32)
                        
                        # Isaac Order: Forward pass through the neural network
                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(obs_batch)
                            action_tensor = self.policy(obs_tensor)
                            if isinstance(action_tensor, tuple):
                                action_tensor = action_tensor[0]
                            action = action_tensor.cpu().numpy().flatten().astype(np.float32)
                        
                        # Parse 12D action: [12 position offsets only]
                        # Update target position (only when policy infers)
                        self.last_action = action.copy()  # Store 12D action for next obs
                        
                        # Position action: 12 dims (relative to default), scale and clip
                        self.qPos_isaac = action[:12] * self.config.action_scale['pos']
                        self.qPos_mj = self.qPos_isaac[self.config.isaac_to_mujoco_map] + self.default_dPos_mj
                        
                        # self.qVel = action[12:24] * self.config.action_scale['vel']  # Not used
                    
                    # PD control: smooth tracking towards target with velocity feedforward (every control step)
                    self.send_command(self.qPos_mj)
                
                # --- Physics Step ---
                self.step()
                motiontime += 1 
                
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


# ========== Main ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sim')
    parser.add_argument('--model', type=str, default='policy.pt')
    parser.add_argument('--config', type=str, default='go1_walk.yaml')
    args = parser.parse_args()
    
    # Adjust config path if it doesn't exist
    if not os.path.exists(args.config):
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(script_dir, 'config', args.config)
    
    # 1. 加载 Config
    config = load_config(args.config)
    
    # 2. 初始化 Gamepad (根据 YAML 中的 gamepad_type 选择手柄)
    gamepad_type = getattr(config, 'gamepad_type', 'f710')
    gamepad = create_gamepad_controller(
        gamepad_type,
        vx_range=config.vx_range,
        vy_range=config.vy_range,
        vyaw_range=config.vyaw_range
    )
    gamepad.start()
    
    print("\n" + "="*70)
    print(f"🎮 Gamepad Control ({gamepad_type})")
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
    
    # Stop gamepad controller
    gamepad.stop()
    print("\nProgram ended.")