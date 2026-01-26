#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Real2Sim: 从真实 Go1 机器人读取关节角度，在 MuJoCo 中实时可视化
用于检验关节顺序映射和电机角度是否正确

使用方法:
1. 确保上位机与 Go1 在同一局域网(192.168.123.x 网段)
2. 运行: python3 real2sim.py --xml ../assets/go1/scene.xml --connect

网络配置:
- Go1 机载电脑 IP: 192.168.123.10 (LowState 数据源)
- 上位机需要配置为 192.168.123.x 网段

SDK 电机索引 -> 训练环境映射:
SDK 顺序:   FR_0, FR_1, FR_2, FL_0, FL_1, FL_2, RR_0, RR_1, RR_2, RL_0, RL_1, RL_2
           (0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11)
训练顺序:   FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...
           (0,     1,        2,       3-5,    6-8,    9-11)
"""

import sys
import time
import math
import numpy as np
import argparse
import mujoco
import mujoco.viewer
import threading
from collections import deque

# 添加 Unitree SDK Python 库路径
sys.path.append('/home/wzh/amp/isaacgym/AMP_for_hardware/unitree_legged_sdk/lib/python/amd64')
try:
    import robot_interface as sdk
    SDK_AVAILABLE = True
except ImportError:
    print("Warning: robot_interface not found. Running in simulation mode.")
    SDK_AVAILABLE = False

# ========== 关节映射常量 ==========

# SDK 中的电机索引（FR, FL, RR, RL 顺序）
SDK_MOTOR_INDEX = {
    'FR_hip': 0, 'FR_thigh': 1, 'FR_calf': 2,
    'FL_hip': 3, 'FL_thigh': 4, 'FL_calf': 5,
    'RR_hip': 6, 'RR_thigh': 7, 'RR_calf': 8,
    'RL_hip': 9, 'RL_thigh': 10, 'RL_calf': 11,
}

# 训练环境中的关节顺序（FL, FR, RL, RR）
TRAINING_JOINT_ORDER = [
    'FL_hip', 'FL_thigh', 'FL_calf',
    'FR_hip', 'FR_thigh', 'FR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf',
]

# SDK 索引 -> 训练环境索引 映射
# SDK: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
# 训练: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
SDK_TO_TRAINING_MAP = [
    3, 4, 5,    # SDK FL(3,4,5) -> Training FL(0,1,2)
    0, 1, 2,    # SDK FR(0,1,2) -> Training FR(3,4,5)
    9, 10, 11,  # SDK RL(9,10,11) -> Training RL(6,7,8)
    6, 7, 8,    # SDK RR(6,7,8) -> Training RR(9,10,11)
]

# MuJoCo 中的关节名称（按训练环境顺序：FL, FR, RL, RR）
JOINT_NAMES = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]

# 默认站立姿态（按训练环境顺序）
DEFAULT_DOF_POS = np.array([
    0.0, 0.9, -1.8,   # FL
    0.0, 0.9, -1.8,   # FR
    0.0, 0.9, -1.8,   # RL
    0.0, 0.9, -1.8    # RR
], dtype=np.float32)


class Real2SimBridge:
    """Real2Sim 数据桥接器：从真实机器人读取数据，传递给 MuJoCo"""
    
    def __init__(self, robot_ip="192.168.123.10", local_port=8080, robot_port=8007):
        """
        初始化 Real2Sim 桥接器
        
        Args:
            robot_ip: Go1 机载电脑 IP (默认 192.168.123.10)
            local_port: 本地 UDP 端口 (默认 8080)
            robot_port: 机器人 UDP 端口 (默认 8007 for LowLevel)
        """
        self.robot_ip = robot_ip
        self.local_port = local_port
        self.robot_port = robot_port
        
        # 状态数据（线程安全）
        self.lock = threading.Lock()
        self.joint_pos = np.zeros(12, dtype=np.float32)   # 关节位置（训练环境顺序）
        self.joint_vel = np.zeros(12, dtype=np.float32)   # 关节速度（训练环境顺序）
        self.imu_quat = np.array([0, 0, 0, 1], dtype=np.float32)  # 四元数 (x,y,z,w)
        self.imu_gyro = np.zeros(3, dtype=np.float32)     # 角速度
        self.imu_rpy = np.zeros(3, dtype=np.float32)      # 欧拉角
        
        # 原始 SDK 数据（用于调试）
        self.raw_joint_pos = np.zeros(12, dtype=np.float32)  # SDK 原始顺序
        
        # 运行状态
        self.running = False
        self.connected = False
        self.last_recv_time = 0
        self.recv_count = 0
        
        # SDK 对象
        self.udp = None
        self.state = None
        
    def connect(self):
        """连接到机器人"""
        if not SDK_AVAILABLE:
            print("SDK not available, using simulation mode")
            self.connected = False
            return False
            
        try:
            LOWLEVEL = 0xff
            # 创建 UDP 连接
            # UDP(uint8_t level, uint16_t localPort, const char* targetIP, uint16_t targetPort)
            self.udp = sdk.UDP(LOWLEVEL, self.local_port, self.robot_ip, self.robot_port)
            self.state = sdk.LowState()
            
            # 初始化命令数据（即使只读取状态，也需要发送一些命令）
            self.cmd = sdk.LowCmd()
            self.udp.InitCmdData(self.cmd)
            
            print(f"UDP connection initialized:")
            print(f"  Local port: {self.local_port}")
            print(f"  Robot IP: {self.robot_ip}:{self.robot_port}")
            
            # 发送初始阻尼模式命令
            print("\nInitializing robot in damping mode...")
            for i in range(12):
                self.cmd.motorCmd[i].mode = 0x00  # 阻尼模式
                self.cmd.motorCmd[i].q = 0
                self.cmd.motorCmd[i].dq = 0
                self.cmd.motorCmd[i].tau = 0
                self.cmd.motorCmd[i].Kp = 0
                self.cmd.motorCmd[i].Kd = 3.0  # 适中阻尼，便于手动移动
            
            # 发送几次初始化命令确保机器人接收
            for _ in range(10):
                self.udp.SetSend(self.cmd)
                self.udp.Send()
                time.sleep(0.002)
            
            print("Robot initialized in damping mode (Kd=3.0)")
            print("You can now manually move the robot joints to verify mapping.\n")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.connected = False
            return False
    
    def _read_state(self):
        """从 SDK 读取一次状态"""
        if not self.connected or self.udp is None:
            return False
            
        try:
            # 接收数据
            self.udp.Recv()
            self.udp.GetRecv(self.state)
            
            # 读取 SDK 原始关节数据（SDK 顺序：FR, FL, RR, RL）
            raw_pos = np.zeros(12, dtype=np.float32)
            raw_vel = np.zeros(12, dtype=np.float32)
            for i in range(12):
                raw_pos[i] = self.state.motorState[i].q
                raw_vel[i] = self.state.motorState[i].dq
            
            # 转换为训练环境顺序（FL, FR, RL, RR）
            training_pos = np.zeros(12, dtype=np.float32)
            training_vel = np.zeros(12, dtype=np.float32)
            for training_idx in range(12):
                sdk_idx = SDK_TO_TRAINING_MAP[training_idx]
                training_pos[training_idx] = raw_pos[sdk_idx]
                training_vel[training_idx] = raw_vel[sdk_idx]
            
            # 读取 IMU 数据
            # SDK 四元数是 (w,x,y,z)，需要转换为 (x,y,z,w)
            quat_wxyz = self.state.imu.quaternion
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
            
            gyro = np.array(self.state.imu.gyroscope, dtype=np.float32)
            rpy = np.array(self.state.imu.rpy, dtype=np.float32)
            
            # 更新状态（线程安全）
            with self.lock:
                self.raw_joint_pos = raw_pos.copy()
                self.joint_pos = training_pos
                self.joint_vel = training_vel
                self.imu_quat = quat_xyzw
                self.imu_gyro = gyro
                self.imu_rpy = rpy
                self.last_recv_time = time.time()
                self.recv_count += 1
            
            return True
            
        except Exception as e:
            print(f"Error reading state: {e}")
            return False
    
    def _send_dummy_cmd(self):
        """发送空命令（保持通信活跃，维持阻尼模式）"""
        if not self.connected or self.udp is None:
            return
            
        try:
            # 设置为阻尼模式（安全），保持与初始化一致
            for i in range(12):
                self.cmd.motorCmd[i].mode = 0x00  # 阻尼模式
                self.cmd.motorCmd[i].q = 0
                self.cmd.motorCmd[i].dq = 0
                self.cmd.motorCmd[i].tau = 0
                self.cmd.motorCmd[i].Kp = 0
                self.cmd.motorCmd[i].Kd = 3.0  # 适中阻尼
            
            self.udp.SetSend(self.cmd)
            self.udp.Send()
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def start(self):
        """启动后台读取线程"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("Real2Sim bridge started")
    
    def stop(self):
        """停止后台线程"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        print("Real2Sim bridge stopped")
    
    def _run_loop(self):
        """后台循环：持续读取机器人状态"""
        dt = 0.002  # 500Hz 读取频率
        while self.running:
            start_time = time.time()
            
            if self.connected:
                self._read_state()
                self._send_dummy_cmd()  # 保持通信
            
            # 控制循环频率
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def get_state(self):
        """获取当前状态（线程安全）"""
        with self.lock:
            return {
                'joint_pos': self.joint_pos.copy(),  # 训练环境顺序
                'joint_vel': self.joint_vel.copy(),
                'raw_joint_pos': self.raw_joint_pos.copy(),  # SDK 原始顺序
                'imu_quat': self.imu_quat.copy(),    # 四元数 (x,y,z,w)
                'imu_gyro': self.imu_gyro.copy(),    # 角速度
                'imu_rpy': self.imu_rpy.copy(),      # 欧拉角 (roll, pitch, yaw)
                'timestamp': self.last_recv_time,
                'recv_count': self.recv_count,
            }


def quat_to_mujoco(quat_xyzw):
    """
    将四元数从 (x,y,z,w) 格式转换为 MuJoCo 的 (w,x,y,z) 格式
    
    Args:
        quat_xyzw: [x, y, z, w] 格式四元数
    
    Returns:
        quat_wxyz: [w, x, y, z] 格式四元数
    """
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def run_real2sim(args):
    """运行 Real2Sim 可视化"""
    
    # ----------------- 加载 MuJoCo 模型 -----------------
    print(f"Loading MuJoCo model: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    
    # 获取关节 qpos 地址
    joint_qpos_addrs = []
    for joint_name in JOINT_NAMES:
        try:
            joint_id = model.joint(joint_name).id
            qpos_addr = model.jnt_qposadr[joint_id]
            joint_qpos_addrs.append(qpos_addr)
            print(f"  {joint_name}: qpos_addr={qpos_addr}")
        except KeyError as e:
            print(f"Error: Joint '{joint_name}' not found in model: {e}")
            raise
    
    # 设置初始姿态
    for i, qpos_addr in enumerate(joint_qpos_addrs):
        data.qpos[qpos_addr] = DEFAULT_DOF_POS[i]
    data.qpos[2] = 0.27  # 基座高度
    
    # 获取基座的 qpos 索引（freejoint: x, y, z, qw, qx, qy, qz）
    # MuJoCo freejoint qpos 布局: [0:3] = position(x,y,z), [3:7] = quaternion(w,x,y,z)
    base_pos_idx = 0  # x, y, z 位置索引
    base_quat_idx = 3  # 四元数起始索引
    
    print(f"Base freejoint indices: pos=[0:3], quat=[3:7]")
    
    # 阻尼模式：将所有控制命令设为 0（无驱动力）
    # MuJoCo motor 执行器 gear=0 时，仅依靠关节阻尼
    for i in range(model.nu):
        data.ctrl[i] = 0.0
    
    mujoco.mj_forward(model, data)
    print(f"MuJoCo initialized in damping mode (all ctrl=0, joint damping active)")
    print(f"Base will sync orientation from robot IMU\n")
    
    # ----------------- 初始化 Real2Sim 桥接器 -----------------
    bridge = Real2SimBridge(
        robot_ip=args.robot_ip,
        local_port=args.local_port,
        robot_port=args.robot_port,
    )
    
    if args.connect:
        if bridge.connect():
            bridge.start()
        else:
            print("Failed to connect to robot. Running in demo mode.")
    else:
        print("Running in demo mode (no robot connection)")
    
    print("\n" + "="*70)
    print("Real2Sim Visualization - Damping Mode")
    print("="*70)
    print("Joint Order Mapping:")
    print("  SDK Order (FR, FL, RR, RL):")
    print("    [0-2: FR_hip/thigh/calf] [3-5: FL] [6-8: RR] [9-11: RL]")
    print("")
    print("  Training Order (FL, FR, RL, RR):")
    print("    [0-2: FL_hip/thigh/calf] [3-5: FR] [6-8: RL] [9-11: RR]")
    print("")
    print("IMU Orientation Mapping:")
    print("  SDK: quaternion (w,x,y,z), rpy (roll, pitch, yaw)")
    print("  MuJoCo base: synced from robot IMU orientation")
    print("  X-axis: forward, Y-axis: left, Z-axis: up")
    print("")
    print("MuJoCo: Damping mode active (motor gear=0, joint damping=1~2)")
    if bridge.connected:
        print("Robot: Connected in damping mode (Kd=3.0)")
        print("Action: Move robot body and joints to verify mapping!")
    else:
        print("Robot: Demo mode (no robot connection)")
    print("="*70)
    
    demo_time = 0.0
    last_print_time = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            
            if bridge.connected and bridge.running:
                # 从真实机器人获取状态
                state = bridge.get_state()
                joint_pos = state['joint_pos']
                imu_quat = state['imu_quat']  # (x, y, z, w)
                
                # 更新 MuJoCo 关节位置
                for i, qpos_addr in enumerate(joint_qpos_addrs):
                    data.qpos[qpos_addr] = joint_pos[i]
                
                # 更新 MuJoCo 基座姿态（从 IMU）
                # 保持 xy 位置不变，只更新高度和姿态
                # data.qpos[0] = 0.0  # x 位置保持
                # data.qpos[1] = 0.0  # y 位置保持
                data.qpos[2] = 0.27  # z 高度固定（或可以根据需要调整）
                
                # 更新基座四元数（转换为 MuJoCo 格式 w,x,y,z）
                quat_mujoco = quat_to_mujoco(imu_quat)
                data.qpos[3:7] = quat_mujoco
                
                # 保持控制命令为 0（阻尼模式）
                for i in range(model.nu):
                    data.ctrl[i] = 0.0
            else:
                # Demo 模式：使用默认姿态 + 简单运动
                demo_time += 0.002
                for i, qpos_addr in enumerate(joint_qpos_addrs):
                    # 简单的呼吸运动
                    offset = 0.1 * np.sin(demo_time * 2.0)
                    if i % 3 == 1:  # thigh joints
                        data.qpos[qpos_addr] = DEFAULT_DOF_POS[i] + offset
                    else:
                        data.qpos[qpos_addr] = DEFAULT_DOF_POS[i]
                
                # 保持控制命令为 0（阻尼模式）
                for i in range(model.nu):
                    data.ctrl[i] = 0.0
            
            # 前向运动学
            mujoco.mj_forward(model, data)
            
            # 更新可视化
            viewer.sync()
            
            # 打印状态（每秒一次）
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                last_print_time = current_time
                
                if bridge.connected and bridge.running:
                    state = bridge.get_state()
                    print(f"\n[Frame: {state['recv_count']}] " + "="*60)
                    
                    # 显示 IMU 姿态信息（重点）
                    print("IMU Orientation:")
                    rpy = state['imu_rpy']
                    print(f"  Roll:  {rpy[0]:+.3f} rad ({np.rad2deg(rpy[0]):+6.1f}°) - 左右倾斜")
                    print(f"  Pitch: {rpy[1]:+.3f} rad ({np.rad2deg(rpy[1]):+6.1f}°) - 前后俯仰")
                    print(f"  Yaw:   {rpy[2]:+.3f} rad ({np.rad2deg(rpy[2]):+6.1f}°) - 左右旋转")
                    
                    quat = state['imu_quat']
                    print(f"  Quaternion (x,y,z,w): [{quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f}]")
                    
                    # 显示 SDK 原始顺序（方便对比真实机器人）
                    print("\nSDK Raw Order (FR, FL, RR, RL):")
                    raw = state['raw_joint_pos']
                    print(f"  FR: [{raw[0]:+.3f}, {raw[1]:+.3f}, {raw[2]:+.3f}]   "
                          f"FL: [{raw[3]:+.3f}, {raw[4]:+.3f}, {raw[5]:+.3f}]")
                    print(f"  RR: [{raw[6]:+.3f}, {raw[7]:+.3f}, {raw[8]:+.3f}]   "
                          f"RL: [{raw[9]:+.3f}, {raw[10]:+.3f}, {raw[11]:+.3f}]")
                    
                    # 显示训练环境顺序（用于 MuJoCo 可视化）
                    print("\nTraining Order (FL, FR, RL, RR):")
                    train = state['joint_pos']
                    print(f"  FL: [{train[0]:+.3f}, {train[1]:+.3f}, {train[2]:+.3f}]   "
                          f"FR: [{train[3]:+.3f}, {train[4]:+.3f}, {train[5]:+.3f}]")
                    print(f"  RL: [{train[6]:+.3f}, {train[7]:+.3f}, {train[8]:+.3f}]   "
                          f"RR: [{train[9]:+.3f}, {train[10]:+.3f}, {train[11]:+.3f}]")
                else:
                    print(f"Demo mode - time: {demo_time:.1f}s")
            
            # 控制循环频率
            elapsed = time.time() - start_time
            if elapsed < 0.002:
                time.sleep(0.002 - elapsed)
    
    # 清理
    if bridge.running:
        bridge.stop()
    print("Real2Sim finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Go1 Real2Sim 可视化')
    parser.add_argument('--xml', type=str, default='/home/wzh/amp/isaacgym/AMP_for_hardware/deploy/assets/go1/scene.xml', help='MuJoCo XML 模型文件路径')
    parser.add_argument('--connect', action='store_true', help='连接到真实机器人')
    parser.add_argument('--robot-ip', type=str, default='192.168.123.10', help='Go1 机载电脑 IP')
    parser.add_argument('--local-port', type=int, default=8080, help='本地 UDP 端口')
    parser.add_argument('--robot-port', type=int, default=8007, help='机器人 UDP 端口')
    args = parser.parse_args()
    
    run_real2sim(args)
