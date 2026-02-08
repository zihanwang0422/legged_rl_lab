# Procedural Quadruped Training Guide

本文档介绍如何使用 metamorphosis 框架在 Isaac Lab 中并行训练任意构型的四足机器人。

## 概述

通过集成 metamorphosis 的程序化机器人生成能力，我们可以在 Isaac Lab 的强化学习框架中训练 **morphology-agnostic**（构型无关）的策略。每个并行环境中的机器人都可以有不同的身体尺寸、腿长等参数。

## 已注册的环境

| 环境 ID | 描述 |
|---------|------|
| `LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-v0` | 平面地形训练环境 |
| `LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-Play-v0` | 平面地形评估环境 |
| `LeggedRLLab-Isaac-Velocity-Rough-Procedural-Quadruped-v0` | 崎岖地形训练环境 |
| `LeggedRLLab-Isaac-Velocity-Rough-Procedural-Quadruped-Play-v0` | 崎岖地形评估环境 |

## 快速开始

### 训练

```bash
# 在平面地形上训练 procedural quadruped
python scripts/rsl_rl/train.py \
    --task=LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-v0 \
    --num_envs 2048 \
    --headless

# 在崎岖地形上训练
python scripts/rsl_rl/train.py \
    --task=LeggedRLLab-Isaac-Velocity-Rough-Procedural-Quadruped-v0 \
    --num_envs 2048 \
    --headless
```

### 评估/可视化

```bash
# 可视化训练好的策略
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-Play-v0 \
    --num_envs 50 \
    --load_run <run_name>
```

## 自定义构型参数范围

可以通过修改 `procedural_env_cfg.py` 中的 `PROCEDURAL_QUADRUPED_CFG` 来调整机器人构型的参数范围：

```python
PROCEDURAL_QUADRUPED_CFG = ArticulationCfg(
    spawn=ProceduralQuadrupedCfg(
        # 身体尺寸范围
        base_length_range=(0.4, 0.8),    # 身体长度 (米)
        base_width_range=(0.2, 0.35),     # 身体宽度 (米)
        base_height_range=(0.1, 0.2),     # 身体高度 (米)
        
        # 腿部尺寸范围
        leg_length_range=(0.5, 0.9),      # 腿长比例
        calf_length_ratio=(0.85, 1.1),    # 小腿/大腿长度比
        
        # 其他参数...
    ),
    # ...
)
```

## 架构说明

### 关键组件

1. **`ProceduralQuadrupedCfg`** (`metamorphosis/src/metamorphosis/asset_cfg.py`)
   - 定义程序化四足机器人的 spawner 配置
   - 使用 `QuadrupedBuilder` 生成不同构型的机器人

2. **`ProceduralRobotEnv`** (`procedural_env.py`)
   - 继承自 `ManagerBasedRLEnv`
   - 在场景初始化后调用 `builder.modify_articulation()` 来调整关节限制

3. **`ProceduralQuadrupedFlatEnvCfg`** (`procedural_env_cfg.py`)
   - 定义训练环境的完整配置
   - 包括奖励函数、终止条件、观测空间等

### 重要注意事项

⚠️ **必须设置 `replicate_physics=False`**

由于每个环境中的机器人构型不同，物理视图是异构的（non-homogeneous），因此必须禁用物理复制：

```python
self.scene.replicate_physics = False
```

## 扩展到其他机器人类型

### 双足机器人 (Biped)

metamorphosis 也支持程序化双足机器人。可以类似地创建：

```python
from metamorphosis.asset_cfg import ProceduralBipedCfg

PROCEDURAL_BIPED_CFG = ArticulationCfg(
    spawn=ProceduralBipedCfg(
        torso_link_length_range=(0.10, 0.16),
        leg_length_range=(0.5, 0.7),
        # ...
    ),
    # ...
)
```

### 轮足机器人 (QuadWheel)

```python
from metamorphosis.asset_cfg import ProceduralQuadWheelCfg

PROCEDURAL_QUADWHEEL_CFG = ArticulationCfg(
    spawn=ProceduralQuadWheelCfg(
        wheel_radius_range=(0.08, 0.15),
        wheel_width_range=(0.03, 0.06),
        # ...
    ),
    # ...
)
```

## 高级用法

### 自定义观测空间

对于构型无关的策略，建议使用归一化的观测：

1. **相对关节位置** (`joint_pos_rel`) - 相对于默认位置的偏移
2. **归一化关节速度** - 使用速度限制进行归一化
3. **机器人形态学参数** - 可以将形态参数作为额外观测

### 添加形态学观测

可以创建自定义的观测函数来包含机器人的形态学信息：

```python
def morphology_params(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回机器人的形态学参数作为观测"""
    from metamorphosis.builder import QuadrupedBuilder
    builder = QuadrupedBuilder.get_instance()
    
    params = torch.tensor([
        [p.base_length, p.base_width, p.thigh_length, p.calf_length]
        for p in builder.params
    ], device=env.device)
    
    return params
```

## 常见问题

### Q: 为什么训练收敛较慢？

A: 构型多样性增加了策略学习的难度。建议：
- 开始时使用较小的构型参数范围
- 使用更大的网络容量
- 启用观测归一化 (`actor_obs_normalization=True`)

### Q: 如何固定某些构型参数？

A: 将参数范围设置为相同的值：

```python
base_length_range=(0.6, 0.6),  # 固定身体长度为 0.6m
```

### Q: 如何验证生成的机器人？

A: 使用 metamorphosis 的独立脚本：

```bash
python metamorphosis/scripts/quadruped_scene.py --num_envs 32
```

## 参考

- [metamorphosis 文档](../metamorphosis/README.md)
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/)
- [rsl_rl 文档](../rsl_rl/README.md)
