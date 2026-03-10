#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Sim2Sim (MuJoCo)

Shared configuration parameters and policy inference logic.

Usage:
    Simulation: python deploy/g1_deploy/sim2sim_walk.py --mode sim --model policy.pt
"""

import time
import numpy as np
import argparse
import torch
import sys
import os
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from joystick import create_gamepad_controller
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import yaml
from pathlib import Path
from types import SimpleNamespace
from collections import deque

LEGGED_RL_LAB_ROOT_DIR = str(Path(__file__).resolve().parent.parent)

# ========== Compute Projected Gravity ==========

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

def build_obs(base_ang_vel, projected_gravity, commands, dof_pos_rel, dof_vel, last_action, config):
    """
    Build the observation vector matching IsaacLab velocity_env_cfg.py PolicyCfg (G1 29 DOF):
    No base_lin_vel in policy obs (only in critic).
    1-3:   base angular velocity (scaled by base_ang_vel)
    4-6:   projected gravity
    7-9:   commands [vx, vy, vyaw]
    10-38: joint position relative to default (29 DOF)
    39-67: joint velocities (scaled by joint_vel, 29 DOF)
    68-96: last actions (29D)
    Total: 96 dims
    """
    obs = []
    
    # 1-3: Base angular velocity (scaled)
    base_ang_vel_scaled = base_ang_vel * config.obs_scales['base_ang_vel']
    obs.extend(list(base_ang_vel_scaled))
    
    # 7-9: Projected gravity
    obs.extend(list(projected_gravity))
    
    # 10-12: Commands [vx, vy, vyaw]
    obs.extend(list(commands))
    
    # 13-24: joint position relative to default
    dof_pos_rel_scaled = dof_pos_rel * config.obs_scales['joint_pos']
    obs.extend(list(dof_pos_rel_scaled))
    
    # 25-36: joint velocities (scaled)
    dof_vel_scaled = dof_vel * config.obs_scales['joint_vel']
    obs.extend(list(dof_vel_scaled))
    
    # 37-48: Last action
    obs.extend(list(last_action))
    
    return np.array(obs, dtype=np.float32)


# ========== Sim2Sim (MuJoCo) controller ==========

class Sim2SimController:
    """MuJoCo simulation controller."""
    
    def __init__(self, config_path, model_name):
        #~/legged_rl_lab/deploy/g1_deploy/
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        # 1. 直接加载并解析配置
        with open(config_path, 'r') as f:
            raw_cfg = yaml.safe_load(f)

        # 2. 极简配置解析器：处理路径映射和类型转换
        self.config = SimpleNamespace(**raw_cfg)
        
        for k, v in raw_cfg.items():
            # 自动转换 List 为 Numpy 数组
            for k, v in raw_cfg.items():
            # 自动转换 List 为 Numpy 数组
                if isinstance(v, list):
                    # --- 关键修改：跳过字符串列表 ---
                    if k in ['joint_names_mujoco', 'actuator_names_mujoco', 'sdk_joint_order']:
                        setattr(self.config, k, v) # 保持原样（字符串列表）
                        continue
                    
                    # 只有数字列表才转换成 numpy 数组
                    try:
                        v = np.array(v, dtype=np.int32 if 'map' in k else np.float32)
                    except (ValueError, TypeError):
                        pass # 如果转换失败（比如列表里混了字符串），保持原样
                
                setattr(self.config, k, v)

        # 3. 基础变量提取
        c = self.config
        xml_filename = os.path.basename(c.xml_path) 
        self.xml_path = os.path.join(_base_dir, "assets", xml_filename)
        self.policy_path = os.path.join(_base_dir, "exported_policy", model_name)

        # 4. 加载 MuJoCo 模型
        print(f"Loading MuJoCo model: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.policy_decimation = int(c.policy_dt / c.sim_dt)
        self.sim_dt = c.sim_dt
        
        # 5. 映射关节与执行器索引 (MuJoCo 顺序)
        # 使用列表推导式精简获取 ID
        self.joint_qpos_addrs = [self.model.jnt_qposadr[self.model.joint(n).id] for n in c.joint_names_mujoco]
        self.joint_qvel_addrs  = [self.model.jnt_dofadr[self.model.joint(n).id] for n in c.joint_names_mujoco]
        self.actuator_ids     = [self.model.actuator(n).id for n in c.actuator_names_mujoco]
        self.num_joints       = len(c.joint_names_mujoco)

        # 6. 加载 Policy
        print(f"Loading policy: {self.policy_path}")
        self.policy = torch.jit.load(self.policy_path, map_location='cpu')
        
        # 7. 初始化缓冲区与增益 (直接从 config 读取)
        self.obs_history = deque([np.zeros(96, dtype=np.float32)] * 5, maxlen=5)
        self.kp = c.kp_walk
        self.kd = c.kd_walk
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        
        # 8. 初始姿态对齐 (Isaac -> MuJoCo)
        # 使用你配置里的 map 数组进行重排
        self.default_qpos_mj = c.default_qpos_isaac[c.isaac_to_mujoco_map]
        self.target_qpos_mj = self.default_qpos_mj.copy()
        
        self.data.qpos[self.joint_qpos_addrs] = self.default_qpos_mj
        self.data.qpos[2] = getattr(c, 'init_height', 0.90) # 优先从配置读高度
        mujoco.mj_step(self.model, self.data)
        
        print("Sim2Sim controller initialized")
        
    def pd_controller(self, target_pos_mj, target_vel_mj):
        """Compute torques via PD control and send to MuJoCo actuators.
        tau = kp * (target_q - current_q) - kd * current_v
        """         
        tau = self.kp * (target_pos_mj - self.data.qpos[self.joint_qpos_addrs]) + self.kd * (target_vel_mj - self.data.qvel[self.joint_qvel_addrs])
        self.data.ctrl[self.actuator_ids] = tau
    
    def run(self, gamepad):
        """
        Main simulation loop with Absolute Time Sync and Render Decimation.
        """
        motiontime = 0 # simulation step counter 
        c = self.config
        
        self.model.opt.timestep = self.sim_dt  # Physics timestep (e.g., 0.005s)

        # Launch viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
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
                
                self.pd_controller(self.target_qpos_mj, np.zeros_like(self.kd))
                
                mujoco.mj_step(self.model, self.data)
                motiontime += 1
                
                if motiontime % self.policy_decimation == 0:
                        
                    quat_wxyz = self.data.qpos[3:7] 
                    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
                    r = R.from_quat(quat_xyzw)    
                    proj_grav = r.apply([0, 0, -1], inverse=True)
                    
                    base_ang = self.data.qvel[3:6]
                    
                    curr_qpos_mj = self.data.qpos[self.joint_qpos_addrs]
                    curr_qvel_mj = self.data.qvel[self.joint_qvel_addrs]
                    
                    curr_qpos_rel_isaac = curr_qpos_mj[c.mujoco_to_isaac_map] - c.default_qpos_isaac
                    curr_qvel_isaac = curr_qvel_mj[c.mujoco_to_isaac_map]
                    # Get user input from gamepad
                    cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                    # cmd_vx, cmd_vy, cmd_vyaw = [0.0, 0.0, 0.4]
                    commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)
                        
                    # Prepare observation for the policy
                    obs = build_obs(base_ang, proj_grav, commands, 
                                        curr_qpos_rel_isaac, curr_qvel_isaac, self.last_action, self.config)
                    self.obs_history.append(obs)
                    # Group-major reorganization (matches training format):
                    # [omega×5, gravity×5, cmd×5, pos×5, vel×5, action×5]
                    obs_arr = np.array(list(self.obs_history))  # (5, 96)
                    n = self.num_joints  # 29
                    obs_input = np.concatenate([
                        obs_arr[:, 0:3].reshape(-1),          # omega × 5 frames
                        obs_arr[:, 3:6].reshape(-1),          # gravity × 5 frames
                        obs_arr[:, 6:9].reshape(-1),          # cmd × 5 frames
                        obs_arr[:, 9:9+n].reshape(-1),        # joint pos × 5 frames
                        obs_arr[:, 9+n:9+2*n].reshape(-1),    # joint vel × 5 frames
                        obs_arr[:, 9+2*n:9+3*n].reshape(-1),  # last action × 5 frames
                    ])
                    obs_batch = obs_input[np.newaxis, :].astype(np.float32)
                        
                    # Isaac Order: Forward pass through the neural network
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(obs_batch)
                        action_tensor = self.policy(obs_tensor)
                        if isinstance(action_tensor, tuple):
                            action_tensor = action_tensor[0]
                        action = action_tensor.cpu().numpy().flatten().astype(np.float32)
                        
                    # Parse 29D action: position offsets
                    self.last_action = action.copy()
                        
                    # Position action: 29 dims (relative to default), scale
                    # Isaac→MuJoCo: result[mujoco_i] = isaac_arr[isaac_to_mujoco_map[mujoco_i]]
                    self.target_qpos_isaac = action * self.config.action_scale['pos'] + c.default_qpos_isaac
                    self.target_qpos_mj = self.target_qpos_isaac[self.config.isaac_to_mujoco_map]
                
                viewer.sync() # Sync rendering to the simulation loop (decimated by render_skip)
                    
                 # Rudimentary time keeping, will drift relative to wall clock.
                expected_real_time = start_time + (motiontime * self.sim_dt)
                time_to_sleep = expected_real_time - time.time()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)   
            
            
                
                # --- Status Telemetry (Original Format) ---
                if motiontime % int(1.0 / self.config.sim_dt) == 0:
                    real_time_now = time.time() - start_time
                    actual_hz = motiontime / real_time_now if real_time_now > 0 else 0
                    elapsed_sim_time = motiontime * self.sim_dt  # use motiontime, not data.time (immune to viewer reset)
                    
                    vx_cur, vy_cur, vyaw_cur = gamepad.get_velocity()
                    
                    print(f"[Gamepad] vx={vx_cur:+.2f} m/s | vy={vy_cur:+.2f} | yaw={vyaw_cur:+.2f} rad/s")
                    print(f"[Sim Time]: t={elapsed_sim_time:.1f}s, Base height: {self.data.qpos[2]:.3f}m")
                    print(f"[Real Time]: t={real_time_now:.1f}s, Actual Hz: {actual_hz:.2f} Hz")


# ========== Main ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='g1_walk.pt')
    parser.add_argument('--config', type=str, default='g1_walk.yaml')
    args = parser.parse_args()
    
    # Adjust config path if it doesn't exist
    if not os.path.exists(args.config):
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(script_dir, 'config', args.config)
    
    # 1. 初始化 Controller (内部加载 Config)
    controller = Sim2SimController(args.config, args.model)

    # 2. 初始化 Gamepad (根据 Config 中的 gamepad_type 选择手柄)
    config = controller.config
    gamepad_type = getattr(config, 'gamepad_type', 'gamesir')
    gamepad = create_gamepad_controller(
        gamepad_type,
        vx_range=config.vx_range,
        vy_range=config.vy_range,
        vyaw_range=config.vyaw_range,
        btn_start=getattr(config, 'gamepad_btn_start', None),
        btn_rb=getattr(config, 'gamepad_btn_rb', None),
        btn_a=getattr(config, 'gamepad_btn_a', None),
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

    # 3. 运行
    controller.run(gamepad)
    
    # Stop gamepad controller
    gamepad.stop()
    print("\nProgram ended.")