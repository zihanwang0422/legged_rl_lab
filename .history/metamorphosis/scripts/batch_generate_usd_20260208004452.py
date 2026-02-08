#!/usr/bin/env python3
# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""
Batch-generate USD files for heterogeneous quadruped training.

This script pre-generates N quadruped USD files with different morphology
parameters offline. Each USD has the SAME topology (12 DOF, same joint names)
but different geometric dimensions.

Usage:
    python batch_generate_usd.py --num_robots 100 --output_dir ./generated_quadrupeds
    python batch_generate_usd.py --num_robots 50 --output_dir ./generated_quadrupeds \
        --base_length_range 0.4 1.0 --leg_length_range 0.5 0.9

The generated files can then be loaded by the PreGeneratedQuadrupedCfg spawner
during training, avoiding expensive online procedural generation.
"""

import argparse
import json
import os
from pathlib import Path

from metamorphosis.builder import QuadrupedBuilder
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf


def make_stage(path: str, meters: float = 1.0, up: str = "Z") -> Usd.Stage:
    """Create a new USD stage with physics scene."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up == "Z" else UsdGeom.Tokens.y)
    return stage


def main():
    parser = argparse.ArgumentParser(description="Batch-generate quadruped USD files")
    parser.add_argument("--num_robots", type=int, default=100, help="Number of USD files to generate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--base_length_range", type=float, nargs=2, default=[0.5, 1.0])
    parser.add_argument("--base_width_range", type=float, nargs=2, default=[0.3, 0.4])
    parser.add_argument("--base_height_range", type=float, nargs=2, default=[0.1, 0.2])
    parser.add_argument("--leg_length_range", type=float, nargs=2, default=[0.5, 0.9])
    parser.add_argument("--calf_length_ratio", type=float, nargs=2, default=[0.85, 1.1])
    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "generated_quadrupeds"
        )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create builder
    builder = QuadrupedBuilder(
        base_length_range=tuple(args.base_length_range),
        base_width_range=tuple(args.base_width_range),
        base_height_range=tuple(args.base_height_range),
        leg_length_range=tuple(args.leg_length_range),
        calf_length_ratio=tuple(args.calf_length_ratio),
    )

    # Store all params for the manifest
    manifest = {
        "num_robots": args.num_robots,
        "generation_config": {
            "base_length_range": args.base_length_range,
            "base_width_range": args.base_width_range,
            "base_height_range": args.base_height_range,
            "leg_length_range": args.leg_length_range,
            "calf_length_ratio": args.calf_length_ratio,
        },
        "robots": [],
    }

    print(f"Generating {args.num_robots} quadruped USD files in: {output_dir}")

    for i in range(args.num_robots):
        usd_filename = f"quadruped_{i:04d}.usda"
        usd_path = str(output_dir / usd_filename)
        
        # Sample parameters with deterministic seed
        param = builder.sample_params(seed=i)
        
        # Create USD stage and spawn robot
        stage = make_stage(usd_path)
        spec = builder.generate_mjspec(param)
        
        # Use the from_mjspec utility to convert MjSpec → USD prim
        from metamorphosis.utils.usd_utils import from_mjspec
        prim = from_mjspec(stage, "/Robot", spec)
        
        # Set as default prim
        stage.SetDefaultPrim(prim)
        stage.GetRootLayer().Save()
        
        # Record morphology parameters
        robot_info = {
            "index": i,
            "filename": usd_filename,
            "params": {
                "base_length": param.base_length,
                "base_width": param.base_width,
                "base_height": param.base_height,
                "thigh_length": param.thigh_length,
                "calf_length": param.calf_length,
                "thigh_radius": param.thigh_radius,
                "parallel_abduction": param.parallel_abduction,
            },
            # Derived quantities useful for normalization
            "derived": {
                "total_leg_length": param.thigh_length + param.calf_length,
                "body_volume": param.base_length * param.base_width * param.base_height,
                "leg_ratio": param.calf_length / param.thigh_length if param.thigh_length > 0 else 1.0,
            },
        }
        manifest["robots"].append(robot_info)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{args.num_robots}] {usd_filename} | "
                  f"base=({param.base_length:.3f}, {param.base_width:.3f}, {param.base_height:.3f}) "
                  f"thigh={param.thigh_length:.3f} calf={param.calf_length:.3f}")

    # Save manifest JSON
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nDone! Generated {args.num_robots} USD files.")
    print(f"Manifest saved to: {manifest_path}")
    print(f"\nMorphology parameter vector dimensions: 7")
    print(f"  [base_length, base_width, base_height, thigh_length, calf_length, thigh_radius, parallel_abduction]")


if __name__ == "__main__":
    main()
