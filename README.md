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

<details>
<summary>Architecture</summary>

| Component | Description |
|---|---|
| `AMPManagerBasedRLEnv` | Overrides `step()` to compute AMP observations **before** terminated environments are reset, so `(s, s')` transition pairs reflect true physics transitions rather than post-reset artifacts. AMP obs are passed via `extras["amp_obs"]`. |
| `AmpRslRlVecEnvWrapper` | Wraps the env for RSL-RL, extracting the `amp` observation group and forwarding it through `extras`. |
| `AMPPPO` | Extends `PPO` with an AMP discriminator, circular replay buffer, and style-reward blending. Discriminator parameters are added to the PPO optimizer as separate param groups with their own learning rate. |
| `AMPDiscriminator` | MLP that takes concatenated `(s, s')` and outputs a logit. Style reward: `r_style = -log(1 - σ(D(s,s'))) = softplus(D(s,s'))`. Trained with BCE loss + R1 gradient penalty + logit regularization. Supports optional empirical observation normalization. |
| `AMPReplayBuffer` | Fixed-size circular buffer storing agent `(state, next_state)` pairs for discriminator training. |
| `MotionLoader` | Loads reference motions from CSV (LAFAN1) or NPZ (AMASS). Automatically reorders joints from MuJoCo/AMASS DFS traversal order to IsaacLab BFS order via a gather map. Supports directory-recursive loading. |

**Reward blending:**

$$r = \alpha \cdot r_{\text{task}} + (1 - \alpha) \cdot r_{\text{style}}$$

where $\alpha$ = `amp_task_reward_lerp` (default 0.4).

**AMP observation features** (per frame, 70-dim for G1):

| Feature | Dim | Source |
|---|---|---|
| `joint_pos_rel` (joint pos - default pos) | 29 | CSV columns / NPZ `dof_positions` |
| `joint_vel` | 29 | Finite diff (CSV) / NPZ `dof_velocities` |
| `base_lin_vel` (base frame) | 3 | Finite diff / NPZ `body_linear_velocities` |
| `base_ang_vel` (base frame) | 3 | Quat diff / NPZ `body_angular_velocities` |
| `foot_pos` (base frame) | 6 | NPZ `body_positions` (CSV: zeros) |

The discriminator input is `history_length=2` consecutive frames concatenated → 140-dim.

</details>

<details>
<summary>Datasets</summary>

**LAFAN1** — Retargeted CSV files (30 FPS). Column layout: `[x, y, z, qx, qy, qz, qw, joint_0 … joint_28]`

```text
LAFAN1_Retargeting_Dataset/
├── g1/               # 40 CSV (all motions combined)
├── g1_walk/          # 12 CSV,  ~86k frames
├── g1_run/           #  4 CSV,  ~28k frames
├── g1_sprint/        #  2 CSV,  ~16k frames
├── g1_dance/         #  8 CSV,  ~45k frames
├── g1_jump/          #  3 CSV,  ~22k frames
├── g1_fall/          #  6 CSV,  ~28k frames
└── g1_fight/         #  5 CSV,  ~36k frames
```

**AMASS** — Retargeted NPZ files from 25 motion-capture databases (17,714 files). Keys: `dof_positions`, `dof_velocities`, `body_positions`, `body_rotations`, `body_linear_velocities`, `body_angular_velocities`. Quaternion convention: `[w, x, y, z]`.

```text
AMASS_Retargeted_for_G1/
└── g1/
    ├── ACCAD/
    ├── BioMotionLab_NTroje/
    ├── CMU/
    ├── DanceDB/
    ├── KIT/
    ├── ...           # 25 sub-databases total
    └── WEIZMANN/
```

</details>

**Registered environments:**

| Environment ID | Description |
|---|---|
| `LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0` | AMP on flat terrain (default motion: AMASS) |
| `LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-Play-v0` | Play/visualize (50 envs, no randomization) |

**Train:**

```bash
# Train with default AMASS dataset
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
    --num_envs 4096 \
    --headless
```

Override motion data via `--motion_file` (supports directory or single file):

```bash
# Train with LAFAN1 walk subset
python scripts/amp/train.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk \
    --num_envs 4096 --headless
```

**Play:**

```bash
python scripts/amp/play.py \
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-Play-v0 \
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk \
    --num_envs 32
```

**Verify Joint Order:**

```bash
python scripts/amp/verify_joint_order.py
```

**Record Reference Motion:**

```bash
python scripts/amp/record_reference_motion.py \
    --task LeggedRLLab-Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --checkpoint logs/rsl_rl/<experiment>/<run>/model_xxx.pt \
    --num_steps 5000 \
    --output source/legged_rl_lab/legged_rl_lab/data/motion/recorded/go2_trot.pt
```

<details>
<summary>Key Hyperparameters</summary>

Defined in `rsl_rl_amp_cfg.py`:

| Parameter | Value | Notes |
|---|---|---|
| `amp_task_reward_lerp` | 0.4 | 40% task / 60% style. Increase for better command tracking |
| `amp_disc_gradient_penalty_coef` | 10.0 | R1 gradient penalty. Increase if discriminator overfits |
| `amp_learning_rate` | 5e-5 | Discriminator LR. Lower than actor/critic to prevent domination |
| `amp_discriminator_hidden_dims` | [1024, 512] | Discriminator MLP architecture |
| `amp_replay_buffer_size` | 1,000,000 | Circular buffer for agent transitions |
| `amp_disc_logit_reg_coef` | 0.05 | L2 regularization on discriminator logits |
| `amp_disc_weight_decay` | 0.0001 | Weight decay for discriminator parameters |

</details>

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

---



## Sim2Sim

Terrain Generator

Use the terrain generator script follow the instruction in [terrain_tool](deploy/utils/terrain_tool/readme.md)

```bash
python3 deploy/utils/terrain_tool/terrain_generator.py
```

<details>
<summary><b>Go2 Walk (Rough)</b></summary>

1. Modify the yaml file in `deploy/go2_deploy/config/go2_walk.yaml`

2. Exported the `policy.pt` to `deploy/go2_deploy/exported_policy`

3. run [sim2sim_walk.py](deploy/go2_deploy/sim2sim_walk.py)


```bash
pip install mujoco
#Walk
python deploy/go2_deploy/sim2sim_walk.py --model go2_rough.pt
```
</details>


<details>
<summary><b>G1 Walk (Rough)</b></summary>
1. Modify the yaml file in `deploy/go2_deploy/config/go2_walk.yaml`

2. Exported the `policy.pt` to `deploy/go2_deploy/exported_policy`

3. run [sim2sim_walk.py](deploy/go2_deploy/sim2sim_walk.py)


```bash
pip install mujoco
#Walk
python deploy/go2_deploy/sim2sim_walk.py --model go2_rough.pt
```
</details>



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


## Acknowledgements

* [robot_lab](https://github.com/fan-ziqi/robot_lab)
* [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab?tab=readme-ov-file#acknowledgements)
* [legged_lab](https://github.com/zitongbai/legged_lab)
* [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
