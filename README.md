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

### Footstand

[<img src="media/footstand_isaac.gif" width="300px">](gifs/isaac.gif)

#### Train

```bash
python scripts/rsl_rl/train.py \
  --task=LeggedRLLab-Isaac-Velocity-Footstand-Unitree-Go2-v0 \
  --num_envs 4096 \
  --headless
```

#### Play

```bash
python scripts/rsl_rl/play.py \
    --task=LeggedRLLab-Isaac-Velocity-Handstand-Unitree-Go2-v0 \
    --num_envs 16
```


</details>

### 🤖️Humanoid

#### AMP (Adversarial Motion Priors)

**Architecture:**
- `AMPManagerBasedRLEnv`: Custom env that captures AMP observations before environment reset
- `AMPPPO`: PPO extended with discriminator, replay buffer, and style reward computation
- `MotionLoader`: Loads LAFAN1 CSV or AMASS NPZ motion data with automatic joint reordering (AMASS DFS -> IsaacLab BFS)

**Supported datasets:**
```text
LAFAN1_Retargeting_Dataset/
├── g1_walk/          # 12 CSV files, 86k frames at 30 FPS
├── g1_run/
├── g1_dance/
├── g1_jump/
├── g1_fall/
└── g1_fight/

AMASS_Retargeted_for_G1/
└── g1/               # NPZ files from multiple motion capture databases
```

**Registered environments:**
| Environment ID | Dataset | Description |
|---|---|---|
| `LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0` | AMASS | General AMP on flat terrain |
| `LeggedRLLab-Isaac-AMP-Walk-Flat-Unitree-G1-v0` | LAFAN1 walk | Walk-specific AMP on flat terrain |

##### Train (LAFAN1 Walk)

```bash
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Walk-Flat-Unitree-G1-v0 \
    --num_envs 4096 \
    --headless
```

You can also override the motion data path:
```bash
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk \
    --headless --num_envs 4096
```

##### Play

```bash
python scripts/amp/play.py \
    --task LeggedRLLab-Isaac-AMP-Walk-Flat-Unitree-G1-Play-v0 \
    --num_envs 32
```

##### Verify Joint Order

To verify that the LAFAN1/AMASS joint order is correctly aligned with IsaacLab:
```bash
python scripts/amp/verify_joint_order.py
```

**Key hyperparameters** (in `rsl_rl_amp_cfg.py`):
- `amp_task_reward_lerp`: 0.4 (40% task, 60% style -- increase for better command tracking)
- `amp_disc_gradient_penalty_coef`: 5.0 (increase if discriminator overfits)
- `amp_discriminator_hidden_dims`: [1024, 512]
- `amp_replay_buffer_size`: 1,000,000

### Metamorphology
#### Train
```bash
python scripts/rsl_rl/train.py     --task LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-v0     --num_envs 4096     --headless
```
#### Play
```bash
python scripts/rsl_rl/play.py     --task LeggedRLLab-Isaac-Velocity-Flat-Procedural-Quadruped-v0     --num_envs 32
```

## Sim2Sim

Define own task:

1. Modify the yaml file in `legged_rl_lab/deploy/config`

2. Exported the `policy.pt` to `legged_rl_lab/deploy/exported_policy`

3. Detail sim2sim guide in  [sim2sim guide](Instruction_CN.md#sim2sim-guide)

4. Play the sim2sim script

```bash
pip install mujoco
#Walk
python deploy/sim2sim_walk.py --mode sim --model policy.pt
#Handstand
python deploy/sim2sim_handstand.py --mode sim --model policy.pt
```

## Sim2Real

Install [unitree_legged_sdk](https://github.com/unitreerobotics/unitree_legged_sdk) for go1:
```bash
git clone https://github.com/unitreerobotics/unitree_legged_sdk.git
```






```bash
python sim2real_walk.py --mode real --model policy.pt
```



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

<!-- 
## Acknowledgements

### rl_locomotion

* [robot_lab](https://github.com/fan-ziqi/robot_lab)
* [basic-locomotion-dls-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-dls-isaaclab)
* [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab?tab=readme-ov-file#acknowledgements)
* [LeggedLab](https://github.com/Hellod035/LeggedLab)
* [parkour_lab](https://github.com/CAI23sbP/Isaaclab_Parkour)
* [wheel_legged_lab](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot)

### AMP/IL_locomotion

* [legged_lab](https://github.com/zitongbai/legged_lab)
* [MimicKit](https://github.com/xbpeng/MimicKit)
* [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP)
* [motion_imitation](https://github.com/erwincoumans/motion_imitation/tree/master)
* [ManagerAMP](https://github.com/XinyuSong123/ManagerAMP)


### motion_tracking_WBC

* [holosoma](https://github.com/amazon-far/holosoma?tab=readme-ov-file)

### loco_mani_WBC

### navigation

* [isaac-go2-ros2](https://github.com/Zhefan-Xu/isaac-go2-ros2)
* [legged-loco](https://github.com/yang-zj1026/legged-loco)
* [go2-ros2](https://github.com/abizovnuralem/go2_omniverse)

### mujoco

* [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
* [mjlab](https://github.com/mujocolab/mjlab)
* [mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
* [FastTD3](https://github.com/younggyoseo/FastTD3) -->