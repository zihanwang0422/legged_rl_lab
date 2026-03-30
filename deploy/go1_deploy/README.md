# Go1 Sim2Sim / Sim2Real 部署

## 目录结构

```
go1_deploy/
├── config/
│   ├── go1_walk.yaml          # 行走策略配置（关节映射、PD 增益、obs scale）
│   └── go1_handstand.yaml     # 倒立策略配置
├── exported_policy/
│   ├── go1_flat.pt            # Flat 行走 JIT 策略
│   ├── go1_rough.pt           # Rough 行走 JIT 策略
│   └── go1_footstand.pt       # 倒立 JIT 策略
├── assets/
│   ├── go1.xml                # Go1 MuJoCo 模型
│   └── scene.xml              # MuJoCo 场景（含地面）
├── sim2sim_walk.py            # Sim2Sim（MuJoCo 仿真验证）
├── sim2real_walk.py           # Sim2Real（真机部署）
├── real2sim.py                # Real2Sim（真机关节→MuJoCo 可视化）
└── unitree_legged_sdk/        # Unitree Go1 低层通信 SDK
```

---

## 1. Sim2Sim（MuJoCo 仿真验证）

在 MuJoCo 中加载策略，用手柄控制，验证策略在仿真中的表现。

### 依赖

```bash
pip install mujoco pyyaml scipy
```

### 运行

```bash
# Flat 行走策略
python deploy/go1_deploy/sim2sim_walk.py --model go1_flat.pt --config go1_walk.yaml

# Rough 行走策略
python deploy/go1_deploy/sim2sim_walk.py --model go1_rough.pt --config go1_walk.yaml
```

### 手柄控制

| 操作 | 功能 |
|---|---|
| 左摇杆 ↑↓ | 前进/后退 (vx) |
| 左摇杆 ←→ | 横移 (vy) |
| 右摇杆 ←→ | 转向 (vyaw) |
| RB + A/B/X/Y | 切换策略槽位 |
| Start | 退出 |

---

## 2. Sim2Real（真机部署）

### 前置条件

1. **编译 unitree_legged_sdk**

   ```bash
   cd deploy/go1_deploy/unitree_legged_sdk
   mkdir build && cd build
   cmake ..
   make
   ```

   编译后 `lib/python/amd64/` 目录下会生成 `robot_interface.so`。

2. **网络连接**
   - 上位机通过网线连接 Go1（LAN 口或机载 NX）
   - 配置 IP 为 `192.168.123.x` 网段
   - Go1 默认 IP: `192.168.123.10`

3. **进入低层控制模式**
   - 遥控器操作：`L2+A` → `L2+A` → `L1+L2+Start`
   - 此时 Go1 电机松弛，等待低层指令

### 运行

```bash
# 在上位机或 Go1 机载 NX 上执行
python deploy/go1_deploy/sim2real_walk.py --model go1_flat.pt --config go1_walk.yaml
```

### 流程

1. **Stand-up**（~2s）：缓慢升高 PD 增益，将关节移到默认站立位
2. **Stabilize**（~0.5s）：以Deploy PD 增益保持站立
3. **Warmup**（~1.6s）：运行策略但发送零速命令，让 obs history 预热
4. **Run**：正式运行，通过 Go1 遥控器控制速度
   - **Ly**（左摇杆上下）→ 前进/后退 vx
   - **Lx**（左摇杆左右）→ 横移 vy
   - **Rx**（右摇杆左右）→ 转向 vyaw
5. **Ctrl+C** 退出 → 自动回到默认站姿


