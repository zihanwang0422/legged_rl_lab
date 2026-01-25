# Manager_based Guide

## Scene & Terrain

## Assets

- [MJCF](https://github.com/google-deepmind/mujoco_menagerie)
- [URDF](https://github.com/robot-descriptions/awesome-robot-descriptions)

https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html

    handstand_feet_height_exp = 10.0
            handstand_feet_on_air = 1.0
            handstand_feet_air_time = 1.0
            handstand_orientation_l2 = -1.0


   if hand_stand_type == "rear":
            air_leg_name = "F.*_foot"  
            feet_height_weight = 10.0
            feet_height = 0.6
            feet_on_air_weight = 10.0
            feet_air_time_weight = 5.0
            target_gravity_weight = -2.5
            target_gravity = [-1.0, 0.0, 0.0]

### Articulation 

下面为robot的物理参数
- [Asset data](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData)

### Convert URDF to USD (recommend)

```bash
cd ~/legged_rl_lab

python scripts/tools/convert_urdf.py   source/legged_rl_lab/legged_rl_lab/data/robots/go1_description/urdf/go1.urdf   source/legged_rl_lab/legged_rl_lab/data/robots/go1_description/usd/go1.usd   --merge-joints   --joint-stiffness 100.0   --joint-damping 0.5   --joint-target-type position
```

### 🚧️ Convert MJCF(.xml) to USD 

https://mujoco.readthedocs.io/en/stable/python.html#usd-exporter

```bash
 python scripts/tools/convert_mjcf.py   source/legged_rl_lab/legged_rl_lab/data/robots/go1_description/mjcf/go1.xml   source/legged_rl_lab/legged_rl_lab/data/robots/go1_description/usd/go1.usd   --import-sites   --make-instanceable
```


## Manager Based Env Design

- [isaaclab.scene](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveSceneCfg)

- [isaaclab.managers](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html)

其继承了manager env中的termcfg
- [isaaclab.envs.mdp](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html)


- [isaaclab.envs.ManagerBasedRLEnv](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html#isaaclab.envs.ManagerBasedRLEnv)
- [综合定义](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)

```python
# mdp/__init__.py
from isaaclab.envs.mdp import *                           # 1. 基础 MDP 函数
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # 2. 官方 velocity MDP
from .curriculums import *                                # 3. 自定义课程学习
from .rewards import *                                    # 4. 自定义奖励
from .terminations import *                               # 5. 自定义终止条件
from .commands import *                                   # 6. 自定义命令
from .observations import *                               # 7. 自定义观察
```

### Rewards

### Observations

### Actions

### Events

### Curriculums

## RL Library

- [isaaclab.rl](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)

- [RL Scripts](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)

- [Add own rl library](https://isaac-sim.github.io/IsaacLab/main/source/how-to/add_own_library.html)

- [Config rl agent](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/configuring_rl_training.html)

- [Train rl agent](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html)


## sim2sim guide

### Convert URDF to USD (recommend)


⚠️ 注意一定要转换成与`go1.xml`和`unitree.py`go1cfg 对应的电机参数`--joint-stiffness 100.0   --joint-damping 0.5   --joint-target-type position`
```bash
cd ~/legged_rl_lab

python scripts/tools/convert_urdf.py   source/legged_rl_lab/legged_rl_lab/data/robots/go1_description/urdf/go1.urdf   source/legged_rl_lab/legged_rl_lab/data/robots/go1_description/usd/go1.usd   --merge-joints   --joint-stiffness 25.0   --joint-damping 1.0   --joint-target-type position
```

### MuJoCo vs. IsaacLab Joint对照表

在`train.py`中加入了打印lab的关节顺序，发现是按照BFS + 正则匹配的关节顺序

```bash
Joint names (in training order):
  [ 0] FL_hip_joint
  [ 1] FR_hip_joint
  [ 2] RL_hip_joint
  [ 3] RR_hip_joint
  [ 4] FL_thigh_joint
  [ 5] FR_thigh_joint
  [ 6] RL_thigh_joint
  [ 7] RR_thigh_joint
  [ 8] FL_calf_joint
  [ 9] FR_calf_joint
  [10] RL_calf_joint
  [11] RR_calf_joint
```



JOINT MAPPING: ISAAC LAB vs. MUJOCO (Unitree Go1)
Isaac Lab Order (Train): Grouped by joint type (Alphabetical within group)
MuJoCo Order (SDK):      Grouped by leg (Kinematic tree / DFS order)

| Train Index | Isaac Lab Joint Name  | MuJoCo Index | MuJoCo Joint Name |
|-------------|-----------------------|--------------|-------------------|
|      0      | FL_hip_joint          |      0       | FL_hip_joint      |
|      1      | FR_hip_joint          |      3       | FR_hip_joint      |
|      2      | RL_hip_joint          |      6       | RL_hip_joint      |
|      3      | RR_hip_joint          |      9       | RR_hip_joint      |
|      4      | FL_thigh_joint        |      1       | FL_thigh_joint    |
|      5      | FR_thigh_joint        |      4       | FR_thigh_joint    |
|      6      | RL_thigh_joint        |      7       | RL_thigh_joint    |
|      7      | RR_thigh_joint        |      10      | RR_thigh_joint    |
|      8      | FL_calf_joint         |      2       | FL_calf_joint     |
|      9      | FR_calf_joint         |      5       | FR_calf_joint     |
|     10      | RL_calf_joint         |      8       | RL_calf_joint     |
|     11      | RR_calf_joint         |      11      | RR_calf_joint     |

MAPPING ARRAYS:
- mujoco_to_isaac_map: [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
  (Use this to reorder MuJoCo sensor data to Isaac Lab format)
  
- isaac_to_mujoco_map: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
  (Use this to reorder Isaac Lab actions to MuJoCo actuator order)

### MuJoCo vs. IsaacLab 电机/关节参数对照表

此处要注意对齐mjcf: [go1.xml](source/legged_rl_lab/legged_rl_lab/data/robots/go1_description/mjcf/go1.xml)和assets中机器人配置文件[unitree.py](source/legged_rl_lab/legged_rl_lab/assets/unitree.py)电机参数

| 物理意义 | 机械描述 | MuJoCo (MJCF) 参数 | IsaacLab (Python) 参数 | Go1 典型值 |
| :--- | :--- | :--- | :--- | :--- |
| **比例增益 (P)** | 关节刚度 (Stiffness) | `kp` (in `<position>`) | `stiffness` | `100.0` |
| **微分增益 (D)** | 关节阻尼 (Damping) | `damping` (in `<joint>`) | `damping` | `0.5 ~ 1.0` |
| **最大扭矩** | 峰值输出力矩 | `forcerange` | `effort_limit` | `23.7 ~ 45.43` |
| **控制限位** | 关节弧度范围 | `ctrlrange` | `ctrl_limit` / `joint_pos` | 见下方说明 |
| **关节摩擦** | 内部干摩擦 | `frictionloss` | `friction` | `0.2` |
| **角速度限制** | 最大旋转速度 | N/A (通常由模型限制) | `velocity_limit` | `30.0 rad/s` |
| **转子惯量** | 电机电枢转动惯量 | `armature` | `armature` | `0.01` |

---

### 关节名称与部位对应关系 (Go1 为例)

| 机器人部位 | 运动学描述 | MuJoCo (MJCF) 类名 | IsaacLab 正则匹配 | 旋转轴 (Axis) |
| :--- | :--- | :--- | :--- | :--- |
| **侧摆关节** | Abduction / Adduction | `abduction` | `.*_hip_joint` | `[1, 0, 0]` |
| **大腿关节** | Hip Flexion / Extension | `hip` | `.*_thigh_joint` | `[0, 1, 0]` |
| **小腿关节** | Knee Flexion / Extension | `knee` | `.*_calf_joint` | `[0, 1, 0]` |

---

