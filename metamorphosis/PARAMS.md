# Metamorphosis 参数修改指南

本文档说明如何调整 `ProceduralQuadrupedCfg` 和 `ProceduralBipedCfg` 的生成参数，
所有参数均为 `(min, max)` 范围，每次 reset 时随机采样一个新形态。

---

## 四足机器人 (QuadrupedBuilder)

### 参数一览

```
base_length_range   ──┐
base_width_range    ──┤  躯干（长方体）
base_height_range   ──┘

leg_length_range    ──── 大腿长度 = base_length × uniform(leg_length_range)
calf_length_ratio   ──── 小腿长度 = 大腿长度 × uniform(calf_length_ratio)

parallel_abduction  ──── [0,1] 概率值：生成 parallel 髋关节的概率
```

### 腿部运动学结构

```
base (躯干)
 └── {FL/FR/RL/RR}_hip     髋外展/内收关节      轴: X
      └── {}_thigh         膝关节（大腿）       轴: Y
           └── {}_calf     踝关节（小腿）       轴: Y
                └── {}_foot  球形接触点
```

### 关键调整场景

| 目标 | 修改参数 |
|------|---------|
| 更大/更重的躯干 | 增大 `base_length_range`, `base_width_range`, `base_height_range` |
| 腿更长 | 增大 `leg_length_range`（相对于躯干长度的比值） |
| 小腿与大腿更接近等长 | 缩小 `calf_length_ratio` 范围并居中于 `1.0` |
| 只生成 parallel 髋 | `parallel_abduction=1.0` |
| 只生成 opposed 髋 | `parallel_abduction=0.0` |

### 示例：紧凑型短腿四足

```python
ProceduralQuadrupedCfg(
    base_length_range=(0.3, 0.5),
    base_width_range=(0.2, 0.3),
    base_height_range=(0.10, 0.18),
    leg_length_range=(0.3, 0.5),   # 腿长约为躯干的 35%~50%
    calf_length_ratio=(0.95, 1.05),
)
```

---

## 双足机器人 (BipedBuilder)

### 层级关系（各参数控制的部位）

```
torso_link          躯干（长方体）
 └── pelvis         腰部连接（圆柱）
      └── {left/right}_hip_pitch_joint  髋pitch段（胶囊）
           └── {}_hip_roll_joint        髋roll段（胶囊）
                └── {}_hip_yaw_joint    大腿（胶囊）
                     └── {}_knee_joint  小腿（胶囊）
                          └── {}_ankle_pitch_link  踝pitch（虚拟体）
                               └── {}_ankle_roll_link  脚掌（长方体）
```

### 参数分组说明

#### 1. 躯干参数
| 参数 | 默认范围 | 说明 |
|------|----------|------|
| `torso_link_length_range` | `(0.10, 0.16)` | 躯干前后长度（m） |
| `torso_link_width_range` | `(0.18, 0.26)` | 躯干左右宽度（m） |
| `torso_link_height_range` | `(0.08, 0.14)` | 躯干上下高度（m）；**不包含腿长** |

> ⚠️ `torso_link_height` 仅是躯干盒子的高度。总站立高度由腰 + 腿决定。

#### 2. 腰部参数
| 参数 | 默认范围 | 说明 |
|------|----------|------|
| `pelvis_height_range` | `(0.05, 0.08)` | 腰部圆柱高度（m） |
| `hip_spacing_range` | `(0.16, 0.24)` | 左右髋关节间距（m） |

#### 3. 髋关节段参数
| 参数 | 默认范围 | 说明 |
|------|----------|------|
| `hip_pitch_link_length_range` | `(0.03, 0.06)` | 髋pitch段长度（m） |
| `hip_pitch_link_radius_range` | `(0.02, 0.04)` | 髋pitch段半径（m） |
| `hip_roll_link_length_range` | `(0.03, 0.06)` | 髋roll段长度（m） |
| `hip_roll_link_radius_range` | `(0.02, 0.04)` | 髋roll段半径（m） |
| `hip_pitch_link_initroll_range` | `(0.0, π/2)` | 髋pitch段初始外旋角（rad），0=竖直，π/2=外展90° |

#### 4. 腿部参数
| 参数 | 默认范围 | 说明 |
|------|----------|------|
| `leg_length_range` | `(0.5, 0.7)` | 大腿+小腿总长（m） |
| `shin_ratio_range` | `(0.85, 1.15)` | 小腿/大腿比值；1.0 = 等长，>1.0 小腿更长 |

脚掌尺寸（`ankle_roll_link`）在 `sample_params` 内部硬编码随机，范围：
- 长度: `(0.18, 0.25)` m
- 宽度: `(0.06, 0.10)` m  
- 高度: `(0.02, 0.03)` m

### 关键调整场景

| 目标 | 修改参数 |
|------|---------|
| 更高的人形机器人 | 增大 `leg_length_range` |
| 更宽髋部（稳定性↑） | 增大 `hip_spacing_range` |
| 腿更笔直（大腿≈小腿） | `shin_ratio_range=(0.95, 1.05)` |
| 小腿明显更长（跳跃型） | `shin_ratio_range=(1.2, 1.5)` |
| 取消外旋髋（G1-like直腿） | `hip_pitch_link_initroll_range=(0.0, 0.1)` |
| 固定躯干大小（减少变化） | min==max，如 `torso_link_height_range=(0.12, 0.12)` |

### 示例：G1 近似尺寸

```python
ProceduralBipedCfg(
    torso_link_length_range=(0.10, 0.16),
    torso_link_width_range=(0.18, 0.26),
    torso_link_height_range=(0.08, 0.14),
    pelvis_height_range=(0.05, 0.08),
    hip_spacing_range=(0.16, 0.24),
    hip_pitch_link_length_range=(0.03, 0.06),
    hip_pitch_link_radius_range=(0.02, 0.04),
    hip_roll_link_length_range=(0.03, 0.06),
    hip_roll_link_radius_range=(0.02, 0.04),
    hip_pitch_link_initroll_range=(0.0, np.pi / 2),
    leg_length_range=(0.5, 0.7),
    shin_ratio_range=(0.85, 1.15),
)
```

---

## 在训练配置中修改参数

训练配置在 `source/legged_rl_lab/.../config/procedural_*/procedural_env_cfg.py` 中，
通过 `PROCEDURAL_HUMANOID_CFG` / `PROCEDURAL_QUADRUPED_CFG` 的 `spawn=` 传入：

```python
PROCEDURAL_HUMANOID_CFG = ArticulationCfg(
    spawn=ProceduralBipedCfg(
        leg_length_range=(0.5, 0.7),   # ← 直接改这里
        shin_ratio_range=(0.85, 1.15),
        ...
    ),
    ...
)
```

修改后直接重新 `train.py` 即可，无需重装包。

---

## .gitignore 说明

`builder/` 目录曾被 `.gitignore` 的 `**/build*/` 规则误忽略（`build*` 匹配 `builder`）。
已在 `.gitignore` 中添加 `!**/builder/` 排除规则修复此问题。
