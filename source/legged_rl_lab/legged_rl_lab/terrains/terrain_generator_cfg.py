"""Pre-defined :class:`TerrainGeneratorCfg` presets for locomotion training.

Each configuration defines a procedural terrain composed of sub-terrains with
varying difficulty.  The ``curriculum`` flag controls whether the terrain
difficulty is managed by a curriculum term during training.

Available presets
-----------------
``GRAVEL_TERRAINS_CFG``
    Low-roughness random terrain — useful as a near-flat baseline.

``ROUGH_TERRAINS_CFG``
    Mixed terrain (stairs, slopes, boxes, waves, pits) for general locomotion.

``DWAQ_TERRAINS_CFG``
    DWAQ-specific progressive terrain — 40 % stairs + 60 % easy terrain so the
    VAE can learn quickly in the early phase of training.

``DWAQ_HARD_TERRAINS_CFG``
    DWAQ late-stage terrain — 70 % narrow stairs (20–28 cm step width) plus
    hard slopes and boxes.  Switch to this via ``resume`` after initial training
    on ``DWAQ_TERRAINS_CFG``.

``STAIRS_TERRAINS_CFG``
    Pure stair terrain (50 % up + 50 % down).

``STAIRS_ONLY_HARD_CFG``
    Pure stairs at maximum difficulty — for play / evaluation only.

``STAIRS_SLOPE_HARD_CFG``
    Mixed stairs + slopes at high difficulty — for play / evaluation.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# ------------------------------------------------------------------ #
#  Near-flat baseline                                                  #
# ------------------------------------------------------------------ #

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
    },
)

# ------------------------------------------------------------------ #
#  General-purpose mixed terrain                                       #
# ------------------------------------------------------------------ #

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs_up_28": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_32": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.1, amplitude_range=(0.0, 0.2), num_waves=5.0
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, inverted=False
        ),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
    },
)

# ------------------------------------------------------------------ #
#  DWAQ: progressive terrain (40% stairs + 60% easy)                   #
# ------------------------------------------------------------------ #

DWAQ_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # stairs up — 20 %
        "stairs_up_26": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_30": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_34": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.06,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # stairs down — 20 %
        "stairs_down_26": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.06,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # easy terrain — 60 %
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.0, 0.02), noise_step=0.01, border_width=0.25
        ),
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.2), platform_width=2.0, inverted=False
        ),
        "rough_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.2), platform_width=2.0, inverted=True
        ),
        "discrete": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.0, 0.1), platform_width=2.0
        ),
    },
)

# ------------------------------------------------------------------ #
#  DWAQ: hard terrain for late-stage resume (70% narrow stairs)        #
# ------------------------------------------------------------------ #

DWAQ_HARD_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # stairs up — 35 %
        "stairs_up_20": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.0, 0.25),
            step_width=0.20,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_24": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.0, 0.25),
            step_width=0.24,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_28": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.11,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # stairs down — 35 %
        "stairs_down_20": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.0, 0.25),
            step_width=0.20,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_24": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.0, 0.25),
            step_width=0.24,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.11,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # other hard terrain — 30 %
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.30, grid_height_range=(0.0, 0.18), platform_width=2.0
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(-0.03, 0.06), noise_step=0.02, border_width=0.25
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.35), platform_width=2.0, inverted=False
        ),
    },
)

# ------------------------------------------------------------------ #
#  Pure stairs (curriculum)                                            #
# ------------------------------------------------------------------ #

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs_up_26": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_30": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_34": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_26": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

# ------------------------------------------------------------------ #
#  Play-only: max-difficulty stairs                                    #
# ------------------------------------------------------------------ #

STAIRS_ONLY_HARD_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(1.0, 1.0),
    sub_terrains={
        "stairs_up_narrow": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.20, 0.25),
            step_width=0.26,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_wide": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.20, 0.25),
            step_width=0.32,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_narrow": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.20, 0.25),
            step_width=0.26,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_wide": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.20, 0.25),
            step_width=0.32,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
    },
)

# ------------------------------------------------------------------ #
#  Play-only: stairs + slopes at high difficulty                       #
# ------------------------------------------------------------------ #

STAIRS_SLOPE_HARD_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(0.8, 1.0),
    sub_terrains={
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.18, 0.25),
            step_width=0.28,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.18, 0.25),
            step_width=0.28,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.25, 0.4), platform_width=2.0, inverted=False
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.25, 0.4), platform_width=2.0, inverted=True
        ),
    },
)
