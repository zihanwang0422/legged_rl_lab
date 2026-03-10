# Deploy

## 环境安装

```bash
pip install -r deploy/requirements.txt
```

---

## G1 Sim2Sim

### 准备

1. 将训练导出的策略文件 `g1_walk.pt` 放入 `g1_deploy/exported_policy/`

2. 按需修改 `g1_deploy/config/g1_walk.yaml`：

   | 字段 | 说明 |
   |------|------|
   | `gamepad_type` | `f710` / `gamesir` / `unitree` |
   | `policy_path` | 策略文件目录 |
   | `xml_path` | MuJoCo XML 路径 |
   | `standup_duration` | 站立保持时长 (s) |
   | `vx/vy/vyaw_range` | 速度指令范围 |
   | `kp_walk` / `kd_walk` | 各关节 PD 增益 (MuJoCo 顺序, 29 DOF) |
   | `default_dPos_isaac` | 默认站立关节角 (Isaac 顺序) |
   | `isaac_to_mujoco_map` / `mujoco_to_isaac_map` | 关节顺序映射 |

### 运行

```bash
cd /path/to/legged_rl_lab
python deploy/g1_deploy/sim2sim_walk.py --model g1_walk.pt
```

### 手柄操作

| 操作 | 功能 |
|------|------|
| **RB + A** | 站立稳定后，按此组合启动 walk policy |
| 左摇杆 上/下 | 前进速度 vx |
| 左摇杆 左/右 | 横移速度 vy |
| 右摇杆 左/右 | 转向速度 vyaw |
| D-pad 上/下 | 目标速度步进 ±0.1 m/s |
| **Start** | 退出程序 |

---

## Go1 Sim2Sim

```bash
python deploy/go1_deploy/sim2sim_walk.py --model go1_walk.pt
```

配置文件：`go1_deploy/config/go1_walk.yaml`，手柄操作同 G1。
