# legged_rl_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

<div align="center">

| <div align="center"> Isaac Lab </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="gifs/isaac.gif" width="200px">](gifs/isaac.gif) | [<img src="gifs/mujoco.gif" width="180px">](gifs/mujoco.gif) | [<img src="gifs/real.gif" width="150px">](gifs/real.gif) |

</div>

## 🧰️Setup 

* Use pip to install isaaclab [pip install isaaclab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html)


* Create conda environment
```bash
    conda create -n env_isaaclab1 python=3.11
    conda activate env_isaaclab1
    pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    pip install --upgrade pip
```

* Install isaacsim 5.1 and isaaclab 2.3
`pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
Verify the installization
`isaacsim`

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

* Run task
```bash
python scripts/rsl_rl/train.py --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go1-v0
``` 

* Play task
```bash
python scripts/rsl_rl/play.py --task=LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go1-v0
```

## Sim2sim

Define own task:

1. Modify the yaml file in `legged_rl_lab/deploy/config`

2. Exported the `policy.pt` to `legged_rl_lab/deploy/exported_policy`

3. Detail sim2sim guide in  [sim2sim guide](Instruction_CN.md#sim2sim-guide)

4. Play the sim2sim script

```bash
python deploy/sim2sim.py --mode sim --model policy.pt
```




## Sim2real
```bash
python sim2sim2real_joystick.py --mode real --model policy_45_continus.pt #play in real-world
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