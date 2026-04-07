from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.sim import schemas
from isaaclab.utils.configclass import configclass
from isaaclab.sim.utils import get_current_stage, find_matching_prim_paths
from typing import Callable

from metamorphosis.builder import QuadrupedBuilder, BipedBuilder, QuadWheelBuilder


def spawn(
    prim_path: str,
    cfg: "ProceduralQuadrupedCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    from pxr import UsdGeom
    
    stage = get_current_stage()
    builder = QuadrupedBuilder(
        base_length_range=cfg.base_length_range,
        base_width_range=cfg.base_width_range,
        base_height_range=cfg.base_height_range,
        leg_length_range=cfg.leg_length_range,
        calf_length_ratio=cfg.calf_length_ratio,
    )

    root_path, asset_path = prim_path.rsplit("/", 1)
    source_prim_paths = find_matching_prim_paths(root_path)
    prim_paths = [
        f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths
    ]
    for i, prim_path in enumerate(prim_paths):
        param = builder.sample_params(seed=i)
        prim = builder.spawn(stage, prim_path, param)
        
        # Compute and set initial height based on this robot's leg parameters
        _, _, standing_height = QuadrupedBuilder._compute_standing_pose(param)
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp().Set((0.0, 0.0, standing_height))
        
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)
    return prim


@configclass
class ProceduralQuadrupedCfg(SpawnerCfg):
    """Configuration parameters for spawning a procedural quadruped."""

    func: Callable = spawn
    """Function to use for spawning the asset."""

    activate_contact_sensors: bool = True
    """Whether to activate contact sensors for the asset. Defaults to True."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None

    visible: bool = True
    """Whether the spawned asset should be visible."""

    semantic_tags: list[tuple[str, str]] | None = None
    """List of semantic tags to add to the spawned asset."""

    copy_from_source: bool = False
    """Whether to copy the asset from the source prim or inherit it."""

    base_length_range: tuple[float, float] = (0.5, 1.0)
    """Range for the base length."""

    base_width_range: tuple[float, float] = (0.3, 0.4)
    """Range for the base width."""

    base_height_range: tuple[float, float] = (0.15, 0.25)
    """Range for the base height."""

    leg_length_range: tuple[float, float] = (0.4, 0.8)
    """Range for the leg length."""

    calf_length_ratio: tuple[float, float] = (0.9, 1.0)
    """Range for the calf length ratio."""


def spawn_biped(
    prim_path: str,
    cfg: "ProceduralBipedCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    stage = get_current_stage()
    builder = BipedBuilder(
        torso_link_length_range=cfg.torso_link_length_range,
        torso_link_width_range=cfg.torso_link_width_range,
        torso_link_height_range=cfg.torso_link_height_range,
        pelvis_height_range=cfg.pelvis_height_range,
        pelvis_radius_coeff_range=cfg.pelvis_radius_coeff_range,
        hip_spacing_range=cfg.hip_spacing_range,
        hip_pitch_link_length_range=cfg.hip_pitch_link_length_range,
        hip_pitch_link_radius_range=cfg.hip_pitch_link_radius_range,
        hip_roll_link_length_range=cfg.hip_roll_link_length_range,
        hip_roll_link_radius_range=cfg.hip_roll_link_radius_range,
        hip_pitch_link_initroll_range=cfg.hip_pitch_link_initroll_range,
        hip_yaw_link_radius_range=cfg.hip_yaw_link_radius_range,
        leg_length_range=cfg.leg_length_range,
        shin_ratio_range=cfg.shin_ratio_range,
        ankle_roll_link_length_range=cfg.ankle_roll_link_length_range,
        ankle_roll_link_width_range=cfg.ankle_roll_link_width_range,
        ankle_roll_link_height_range=cfg.ankle_roll_link_height_range,
        head_radius_range=cfg.head_radius_range,
        arm_length_range=cfg.arm_length_range,
        forearm_ratio_range=cfg.forearm_ratio_range,
        upper_arm_radius_range=cfg.upper_arm_radius_range,
        torso_link_mass_range=cfg.torso_link_mass_range,
        hip_pitch_link_mass_range=cfg.hip_pitch_link_mass_range,
        hip_roll_link_mass_range=cfg.hip_roll_link_mass_range,
        hip_yaw_link_mass_coeff_range=cfg.hip_yaw_link_mass_coeff_range,
        knee_link_radius_coeff_range=cfg.knee_link_radius_coeff_range,
        knee_link_mass_coeff_range=cfg.knee_link_mass_coeff_range,
        ankle_roll_link_mass_range=cfg.ankle_roll_link_mass_range,
        head_mass_coeff_range=cfg.head_mass_coeff_range,
        upper_arm_mass_coeff_range=cfg.upper_arm_mass_coeff_range,
        forearm_radius_coeff_range=cfg.forearm_radius_coeff_range,
        forearm_mass_coeff_range=cfg.forearm_mass_coeff_range,
    )

    root_path, asset_path = prim_path.rsplit("/", 1)
    source_prim_paths = find_matching_prim_paths(root_path)
    prim_paths = [
        f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths
    ]
    for i, prim_path in enumerate(prim_paths):
        param = builder.sample_params(seed=i)
        prim = builder.spawn(stage, prim_path, param)
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)
    return prim


@configclass
class ProceduralBipedCfg(SpawnerCfg):
    """Configuration parameters for spawning a procedural biped (G1-style)."""

    func: Callable = spawn_biped

    activate_contact_sensors: bool = True
    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None
    visible: bool = True
    semantic_tags: list[tuple[str, str]] | None = None
    copy_from_source: bool = False

    # \u2500\u2500 Geometry ranges \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    torso_link_length_range:        tuple[float, float] = (0.10, 0.16)
    torso_link_width_range:         tuple[float, float] = (0.18, 0.26)
    torso_link_height_range:        tuple[float, float] = (0.20, 0.30)
    pelvis_height_range:            tuple[float, float] = (0.05, 0.08)
    pelvis_radius_coeff_range:      tuple[float, float] = (0.20, 0.40)   # \u00d7 min(w, l)
    hip_spacing_range:              tuple[float, float] = (0.16, 0.24)
    hip_pitch_link_length_range:    tuple[float, float] = (0.03, 0.06)
    hip_pitch_link_radius_range:    tuple[float, float] = (0.02, 0.04)
    hip_roll_link_length_range:     tuple[float, float] = (0.03, 0.06)
    hip_roll_link_radius_range:     tuple[float, float] = (0.02, 0.04)
    hip_pitch_link_initroll_range:  tuple[float, float] = (0.00, 0.20)   # rad ~0-11\u00b0
    hip_yaw_link_radius_range:      tuple[float, float] = (0.025, 0.040)
    leg_length_range:               tuple[float, float] = (0.50, 0.70)
    shin_ratio_range:               tuple[float, float] = (0.85, 1.15)
    ankle_roll_link_length_range:   tuple[float, float] = (0.18, 0.25)
    ankle_roll_link_width_range:    tuple[float, float] = (0.06, 0.10)
    ankle_roll_link_height_range:   tuple[float, float] = (0.02, 0.03)
    head_radius_range:              tuple[float, float] = (0.06, 0.12)
    arm_length_range:               tuple[float, float] = (0.20, 0.40)
    forearm_ratio_range:            tuple[float, float] = (0.80, 1.10)
    upper_arm_radius_range:         tuple[float, float] = (0.020, 0.040)

    # \u2500\u2500 Mass / coefficient ranges \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    torso_link_mass_range:          tuple[float, float] = (3.0,  5.0)
    hip_pitch_link_mass_range:      tuple[float, float] = (0.4,  0.8)
    hip_roll_link_mass_range:       tuple[float, float] = (0.4,  0.8)
    hip_yaw_link_mass_coeff_range:  tuple[float, float] = (1.5,  2.2)    # \u00d7 length
    knee_link_radius_coeff_range:   tuple[float, float] = (0.75, 0.95)   # \u00d7 hip_yaw_radius
    knee_link_mass_coeff_range:     tuple[float, float] = (1.2,  1.8)    # \u00d7 length
    ankle_roll_link_mass_range:     tuple[float, float] = (0.2,  0.5)
    head_mass_coeff_range:          tuple[float, float] = (250.0, 450.0) # \u00d7 radius\u00b3
    upper_arm_mass_coeff_range:     tuple[float, float] = (1.0,  2.0)    # \u00d7 length
    forearm_radius_coeff_range:     tuple[float, float] = (0.75, 0.95)   # \u00d7 ua_radius
    forearm_mass_coeff_range:       tuple[float, float] = (0.8,  1.5)    # \u00d7 length


def spawn_quadwheel(
    prim_path: str,
    cfg: "ProceduralQuadWheelCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    stage = get_current_stage()
    builder = QuadWheelBuilder(
        base_length_range=cfg.base_length_range,
        base_width_range=cfg.base_width_range,
        base_height_range=cfg.base_height_range,
        leg_length_range=cfg.leg_length_range,
        calf_length_ratio=cfg.calf_length_ratio,
        wheel_radius_range=cfg.wheel_radius_range,
        wheel_width_range=cfg.wheel_width_range,
    )

    root_path, asset_path = prim_path.rsplit("/", 1)
    source_prim_paths = find_matching_prim_paths(root_path)
    prim_paths = [
        f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths
    ]
    for i, prim_path in enumerate(prim_paths):
        param = builder.sample_params(seed=i)
        prim = builder.spawn(stage, prim_path, param)
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(prim_path, stage=stage)
    return prim


@configclass
class ProceduralQuadWheelCfg(SpawnerCfg):
    """Configuration parameters for spawning a procedural quad-wheel robot."""

    func: Callable = spawn_quadwheel
    """Function to use for spawning the asset."""

    activate_contact_sensors: bool = True
    """Whether to activate contact sensors for the asset. Defaults to True."""

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None

    visible: bool = True
    """Whether the spawned asset should be visible."""

    semantic_tags: list[tuple[str, str]] | None = None
    """List of semantic tags to add to the spawned asset."""

    copy_from_source: bool = False
    """Whether to copy the asset from the source prim or inherit it."""

    base_length_range: tuple[float, float] = (0.5, 1.0)
    """Range for the base length."""

    base_width_range: tuple[float, float] = (0.3, 0.4)
    """Range for the base width."""

    base_height_range: tuple[float, float] = (0.15, 0.25)
    """Range for the base height."""

    leg_length_range: tuple[float, float] = (0.4, 0.8)
    """Range for the leg length."""

    calf_length_ratio: tuple[float, float] = (0.9, 1.0)
    """Range for the calf length ratio."""

    wheel_radius_range: tuple[float, float] = (0.08, 0.15)
    """Range for the wheel radius in meters."""

    wheel_width_range: tuple[float, float] = (0.03, 0.06)
    """Range for the wheel width in meters."""
