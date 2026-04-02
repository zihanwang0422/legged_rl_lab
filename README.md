# legged_rl_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

<div style="margin: auto; width: fit-content;">

<table border="1">
  <tr>
    <th>Robot</th>
    <th>Task</th>
    <th>Sim2Sim</th>
    <th>Sim2Real</th>
  </tr>
  <tr>
    <td rowspan="3">Unitree Go1</td>
    <td>Flat</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>Rough</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Footstand</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>OpenDuck Mini</td>
    <td>Flat</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">Motion Tracking</td>
    <td>G1 (Tracking)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>SMPL Humanoid (Tracking)</td>
    <td></td>
    <td></td>
  </tr>
</table>

</div>



## 🧰️Setup 

* Use pip to install isaaclab [pip install isaaclab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html)


* Create conda environment
```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade pip
```

* Install isaacsim 5.1 and isaaclab 2.3
```bash
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
```
Verify the installization
```bash
isaacsim
```

* Install the project
```bash
python -m pip install -e source/legged_rl_lab
```

* List the tasks available in the project
```bash
python scripts/list_envs.py
```

---

## 🚀Train

### 🐕️Go2

<details>
<summary><b>Walk (Flat)</b></summary>

#### Walk (Flat)

[<img src="media/walkflat_isaac.gif" width="300px">](gifs/isaac.gif)


```bash 
#Train
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Flat-Unitree-Go1-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run /path/to/log/folder \
  --checkpoint model_xx.pt  
```

```bash
#Play
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Flat-Unitree-Go1-v0 \
    --num_envs 16
```


</details>

<details>
<summary><b>Walk (Rough)</b></summary>

#### Walk(rough)

[<img src="media/walkrough_isaac.gif" width="300px">](gifs/walkrough.gif)

```bash
#Train
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 4096 \
  --headless

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=54321 \
    scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --num_envs 4096 \
    --headless  
```

```bash
#Play
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go1-v0 \
    --num_envs 16
```


</details>

<details>
<summary><b>Handstand</b></summary>

#### Footstand

[<img src="media/footstand_isaac.gif" width="300px">](gifs/isaac.gif)


```bash
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Footstand-Unitree-Go2-v0 \
  --num_envs 4096 \
  --headless
```

```bash
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Handstand-Unitree-Go2-v0 \
    --num_envs 16
```


</details>

### 🤖️G1

<details>
<summary><b>Walk (Flat)</b></summary>

```bash
#Train
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Flat-Unitree-G1-v0 \
  --num_envs 4096 \
  --headless
```

```bash
#Play
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Flat-Unitree-G1-v0 \
    --num_envs 16
```

</details>

<details>
<summary><b>Walk (Rough)</b></summary>

```bash
#Train
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-G1-v0 \
  --num_envs 4096 \
  --headless
```

```bash
#Play
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-G1-v0 \
    --num_envs 16
```

</details>

<details>
<summary><b>AMP (Adversarial Motion Priors)</b></summary>

> 算法原理、架构设计、数据集重定向方法与关节对应关系详见 [Instruction_CN.md](Instruction_CN.md#amp-in-isaaclab)

**数据集**

将以下数据集放入对应目录：

```
source/legged_rl_lab/legged_rl_lab/data/motion/
├── LAFAN1_Retargeting_Dataset/   # 动作捕捉重定向 CSV（30 FPS）
│   ├── g1_walk/                  # 12 CSV, ~86k frames
│   ├── g1_run/                   #  4 CSV, ~28k frames
│   ├── g1_sprint/                #  2 CSV, ~16k frames
│   ├── g1_dance/                 #  8 CSV, ~45k frames
│   ├── g1_jump/                  #  3 CSV, ~22k frames
│   ├── g1_fall/                  #  6 CSV, ~28k frames
│   └── g1_fight/                 #  5 CSV, ~36k frames
└── AMASS_Retargeted_for_G1/      # 大规模动捕 NPZ（25 个子库，17,714 文件）
    └── g1/
        ├── CMU/
        ├── KIT/
        └── ...
```

- LAFAN1 重定向数据：[LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset)
- AMASS 重定向数据：[AMASS_Retargeted_for_G1](https://huggingface.co/datasets/unitreerobotics/AMASS_Retargeted_for_G1)

```bash
# Train with AMASS (default)
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
    --num_envs 4096 \
    --headless

# Train with LAFAN1 walk
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk \
    --num_envs 4096 --headless

# Resume training
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk \
    --resume --load_run <run_folder> --checkpoint model_xxx.pt \
    --num_envs 4096 --headless

python -m torch.distributed.run \
  --nproc_per_node=4 \
  scripts/amp/train.py \
  --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
  --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk \
  --num_envs 4096 --headless \
  --distributed  
```

```bash
#Play
python scripts/amp/play.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-Play-v0 \
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk \
    --num_envs 32
```

</details>

<details>
<summary><b>Procedural (Metamorphosis)</b></summary>

```bash
#Train
python scripts/rsl_rl/train.py \
    --task LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-v0 \
    --num_envs 4096 \
    --headless
```

```bash
#Play
python scripts/rsl_rl/play.py \
    --task LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-v0 \
    --num_envs 32
```

</details>

<details>
<summary><b>Cross-Embodied (G1 + Go2)</b></summary>

```bash
#Train (multi-GPU)
python -m torch.distributed.run \
  --nproc_per_node=4 \
  scripts/rsl_rl/train_cross_embodied_shared.py \
  --num_envs 4096 \
  --headless

#Train (single-GPU)
python scripts/rsl_rl/train_cross_embodied_shared.py \
  --num_envs 4096 \
  --headless
```

```bash
#Play
python scripts/rsl_rl/play_cross_embodied_shared.py \
  --num_envs 32
```

</details>

### 🏃 Motion Tracking

**Dataset**: [LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset) (CSV, 30 FPS, retargeted to G1-29DOF)

```bash
# Step 1 — Convert retargeted CSV to NPZ (runs FK via Isaac Sim to compute full body states)
python scripts/csv_to_npz.py \
  --input_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_fall/fallAndGetUp1_subject1.csv \
  --input_fps 30 \
  --headless
```

```bash
# Step 2 — (Optional) Replay NPZ in Isaac Sim to verify
python scripts/replay_npz.py \
    --file /path/to/npz_file
```

```bash
# Step 3 — Train
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /path/to/motion.npz \
  --num_envs 4096 --headless

# Resume
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /path/to/motion.npz \
  --resume --load_run <run_folder> --checkpoint model_xxx.pt \
  --num_envs 4096 --headless
```

```bash
# Step 4 — Play
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /path/to/motion.npz \
  --num_envs 16
```
[<img src="media/mimic_lafan.gif" width="300px">](gifs/walkrough.gif)

| Task ID | Description |
|---------|-------------|
| `Tracking-Flat-G1-v0` | Standard, with state estimation |
| `Tracking-Flat-G1-Wo-State-Estimation-v0` | No state estimation (closer to real deployment) |
| `Tracking-Flat-G1-Low-Freq-v0` | Half-frequency control |

---



## Sim2Sim

Terrain Generator: use the terrain generator script, see [terrain_tool](deploy/utils/terrain_tool/readme.md) for details.

```bash
python3 deploy/utils/terrain_tool/terrain_generator.py
```

<details>
<summary><b>Go1 Walk</b></summary>

详细说明见 [deploy/go1_deploy/README.md](deploy/go1_deploy/README.md)

```bash
pip install mujoco
python deploy/go1_deploy/sim2sim_walk.py --model go1_flat.pt
```

</details>

<details>
<summary><b>Go2 Walk / Handstand</b></summary>

详细说明见 [deploy/go2_deploy/README.md](deploy/go2_deploy/README.md)

```bash
pip install mujoco
# Walk
python deploy/go2_deploy/sim2sim_walk.py --model go2_rough.pt
# Handstand
python deploy/go2_deploy/sim2sim_handstand.py --model go2_handstand.pt
```

</details>

<details>
<summary><b>G1 Walk</b></summary>

详细说明见 [deploy/g1_deploy/README.md](deploy/g1_deploy/README.md)

```bash
pip install mujoco
python deploy/g1_deploy/sim2sim_walk.py --model g1_flat.pt --config g1_walk.yaml
```

</details>

---

## Sim2Real

<details>
<summary><b>Go1 Walk</b></summary>

详细说明见 [deploy/go1_deploy/README.md](deploy/go1_deploy/README.md)

```bash
# 依赖：unitree_legged_sdk（见 README）
python deploy/go1_deploy/sim2real_walk.py --mode real --model policy.pt
```

</details>

<details>
<summary><b>Go2 Walk</b></summary>

详细说明见 [deploy/go2_deploy/README.md](deploy/go2_deploy/README.md)

```bash
python deploy/go2_deploy/sim2real_walk.py --mode real --model policy.pt
```

</details>

<details>
<summary><b>G1 Walk</b></summary>

详细说明见 [deploy/g1_deploy/README.md](deploy/g1_deploy/README.md)

```bash
# 依赖：cyclonedds + unitree_sdk2_python（见 README）
python deploy/g1_deploy/sim2real_walk.py
```

</details>



## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/legged_rl_lab"
    ]
}
```

### Restart Terminal
```bash
pkill -f "python.*train.py"
```


## Acknowledgements

* [robot_lab](https://github.com/fan-ziqi/robot_lab)
* [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab?tab=readme-ov-file#acknowledgements)
* [legged_lab](https://github.com/zitongbai/legged_lab)
* [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
