with open("README.md", "r") as f:
    text = f.read()

import re

amp_section = """### AMP
dataset:
[Lafan](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
[AMASS](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1)

The Adversarial Motion Priors (AMP) tasks allow imitating reference datasets (like walking, running, crouching) dynamically.

#### Train

To specify a dataset folder or a specific motion file, use the `--motion_file` argument.
For example, to train Unitree G1 to reproduce Lafan walking traits:
```bash
python scripts/amp/train.py \\
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-v0 \\
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk \\
    --headless --num_envs=4096
```

#### Play

To visualize a trained AMP model naturally recreating movements smoothly:
```bash
python scripts/amp/play.py \\
    --task LeggedRLLab-Isaac-AMP-Flat-Unitree-G1-Play-v0 \\
    --motion_file source/legged_rl_lab/legged_rl_lab/data/motion/LAFAN1_Retargeting_Dataset/g1_walk \\
    --num_envs=32
```
"""

text = re.sub(r'### AMP.*?#### Play\n+((```)|)', amp_section, text, flags=re.DOTALL)

with open("README.md", "w") as f:
    f.write(text)
