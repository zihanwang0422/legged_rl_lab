# metamorphosis

A minimalist implementation of morphology randomization in Isaac Sim. Can be used with or without IsaacLab.

## TODO:

- [x] Preliminary implementation.
- [x] `QuadrupedBuilder12Dof` and `QuadrupedBuilder16Dof`.
- [ ] A minimal RL baseline.  

## Installation

Git clone and `pip install -e .`.

## Setup

See `scripts/`.

Note that running independently (without IsaacLab) would require installing Open-USD via `pip install usd-core`.
However, IsaacSim uses a custom USD build and is **incompatible** with the one installed via pip.
If you experience IsaacSim errors after `pip install usd-core`, uninstall it.


## Generate Robots

```python
#QuadrupedBuilder12Dof
python scripts/quadruped_scene.py --num_envs 32 

#QuadrupedBuilder16Dof(quadwheel)
python scripts/quawheel_scene.py

#BipedBuilder
python scripts/biped_scene.py

```

**Important**

Remember to set `replicate_physics=False` and `enabled_self_collisions=False`!

## Citation
If you use this codebase in your research, please cite:

```
@misc{metamorphosis2026,
  title={Metamorphosis: A Framework for Procedural Asset Generation in Isaac Sim},
  author={Botian Xu},
  year={2026},
  url={https://github.com/btx0424/metamorphosis}
}
```

