# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Pre-generated USD Spawner for heterogeneous quadruped training.

This module provides a spawner that loads pre-generated USD files with
different morphologies for each environment instance. All robots share
the same topology (12 DOF, same joint names) but differ in geometric
parameters.

Key design:
    - USD files are generated offline via batch_generate_usd.py
    - A manifest.json records the morphology parameters for each file
    - The spawner assigns env_index % num_files to each environment
    - Morphology parameters are cached for injection into observations
"""

from __future__ import annotations

import json
import os
from typing import Callable

import torch
import numpy as np

from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.sim import schemas
from isaaclab.sim.utils import find_matching_prim_paths
from isaaclab.utils.configclass import configclass

from pxr import Usd, UsdGeom, Sdf


def _spawn_from_pregenerated_usd(
    prim_path: str,
    cfg: "PreGeneratedQuadrupedCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn heterogeneous quadrupeds from pre-generated USD files.
    
    For each environment instance, loads a different USD file based on
    (env_index % num_usd_files). All USD files share the same topology
    (identical joint names and structure) but differ in geometry.
    
    Also populates cfg._morphology_params with per-env parameter vectors
    for later injection into observations.
    
    Args:
        prim_path: The prim path pattern (e.g., "/World/envs/env_.*/Robot").
        cfg: The spawner configuration.
        translation: Optional translation for the root prim.
        orientation: Optional orientation for the root prim.
    
    Returns:
        The last spawned prim.
    """
    # Load manifest
    manifest_path = os.path.join(cfg.usd_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run batch_generate_usd.py first to generate USD files."
        )
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    num_usd_files = len(manifest["robots"])
    if num_usd_files == 0:
        raise ValueError("No robots found in manifest. Run batch_generate_usd.py first.")
    
    # Parse prim paths
    root_path, asset_path = prim_path.rsplit("/", 1)
    source_prim_paths = find_matching_prim_paths(root_path)
    prim_paths = [
        f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths
    ]
    
    num_envs = len(prim_paths)
    
    # Prepare morphology parameter storage
    # Parameter vector: [base_length, base_width, base_height, thigh_length, calf_length, thigh_radius, parallel_abduction]
    morphology_params = np.zeros((num_envs, 7), dtype=np.float32)
    
    # Cache USD references to avoid repeated file reads
    usd_references = {}
    
    prim = None
    for i, target_prim_path in enumerate(prim_paths):
        # Select USD file for this env
        usd_idx = i % num_usd_files
        robot_info = manifest["robots"][usd_idx]
        usd_filename = robot_info["filename"]
        usd_filepath = os.path.join(cfg.usd_dir, usd_filename)
        
        if not os.path.exists(usd_filepath):
            raise FileNotFoundError(f"USD file not found: {usd_filepath}")
        
        # Get absolute path for USD reference
        usd_abs_path = os.path.abspath(usd_filepath)
        
        # Add as USD reference on the target prim path
        stage = Usd.Stage.Open(usd_abs_path)
        default_prim = stage.GetDefaultPrim()
        
        # Get the current (target) stage
        from isaaclab.sim.utils import get_current_stage
        target_stage = get_current_stage()
        
        # Define the prim and add reference
        target_prim = target_stage.DefinePrim(target_prim_path)
        target_prim.GetReferences().AddReference(usd_abs_path)
        
        # Apply articulation root properties
        if cfg.articulation_props is not None:
            schemas.modify_articulation_root_properties(target_prim_path, cfg.articulation_props)
        
        # Activate contact sensors if requested
        if cfg.activate_contact_sensors:
            schemas.activate_contact_sensors(target_prim_path, stage=target_stage)
        
        prim = target_prim
        
        # Store morphology parameters
        params = robot_info["params"]
        morphology_params[i] = [
            params["base_length"],
            params["base_width"],
            params["base_height"],
            params["thigh_length"],
            params["calf_length"],
            params["thigh_radius"],
            float(params["parallel_abduction"]),
        ]
    
    # Store params on the cfg so the environment class can retrieve them
    cfg._morphology_params = morphology_params
    cfg._manifest = manifest
    cfg._num_usd_files = num_usd_files
    
    print(f"[PreGeneratedQuadruped] Loaded {num_envs} envs from {num_usd_files} USD files")
    print(f"[PreGeneratedQuadruped] Morphology param shape: {morphology_params.shape}")
    
    return prim


@configclass
class PreGeneratedQuadrupedCfg(SpawnerCfg):
    """Configuration for spawning from pre-generated USD files.
    
    Usage:
        1. First run batch_generate_usd.py to generate USD files + manifest.json
        2. Set usd_dir to the directory containing the generated files
        3. Each env instance loads a different USD (env_index % num_files)
        4. Morphology parameters are automatically cached for observation injection
    
    Attributes:
        usd_dir: Directory containing pre-generated .usda files and manifest.json.
        activate_contact_sensors: Whether to enable contact sensors.
        articulation_props: Articulation root properties to apply.
    """
    
    func: Callable = _spawn_from_pregenerated_usd
    """Function to use for spawning the asset."""
    
    activate_contact_sensors: bool = True
    """Whether to activate contact sensors for the asset."""
    
    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None
    """Articulation root properties."""
    
    visible: bool = True
    """Whether the spawned asset should be visible."""
    
    semantic_tags: list[tuple[str, str]] | None = None
    """List of semantic tags to add to the spawned asset."""
    
    copy_from_source: bool = False
    """Whether to copy the asset from the source prim or inherit it."""
    
    usd_dir: str = ""
    """Directory containing pre-generated USD files and manifest.json."""
    
    # Runtime-populated fields (do NOT set manually)
    _morphology_params: np.ndarray | None = None
    """Per-env morphology parameters [N, 7]. Populated by spawn function."""
    
    _manifest: dict | None = None
    """Loaded manifest data. Populated by spawn function."""
    
    _num_usd_files: int = 0
    """Number of available USD files. Populated by spawn function."""
