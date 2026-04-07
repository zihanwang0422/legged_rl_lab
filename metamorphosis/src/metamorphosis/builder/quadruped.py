from __future__ import annotations

import torch
import mujoco
import numpy as np
import random

from typing import NamedTuple, Callable, TYPE_CHECKING
from metamorphosis.builder.base import BuilderBase
from metamorphosis.utils.mjs_utils import add_capsule_geom_

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


class QuadrupedParam(NamedTuple):
    base_length: float
    base_width: float
    base_height: float
    thigh_length: float
    calf_length: float
    thigh_radius: float
    parallel_abduction: bool


class QuadrupedBuilder(BuilderBase):
    """
    Builder for generating procedurally parameterized quadruped robots.
    
    This class creates quadruped robot models with configurable body dimensions
    and leg proportions. It samples random parameters within specified ranges
    and generates MuJoCo specifications for the robot model.
    
    The quadruped consists of:
    - A base body (trunk) with configurable length, width, and height
    - Four legs, each with:
      - A hip joint (abduction/adduction)
      - A thigh segment with a knee joint
      - A calf segment
      - A foot (spherical contact point)
    
    Attributes:
        base_length_range: Tuple of (min, max) for base body length in meters.
        base_width_range: Tuple of (min, max) for base body width in meters.
        base_height_range: Tuple of (min, max) for base body height in meters.
        leg_length_range: Tuple of (min, max) for leg length as a ratio of base_length.
        calf_length_ratio: Tuple of (min, max) for calf length as a ratio of thigh_length.
        parallel_abduction: Whether to use parallel abduction/adduction (Unitree Go2 and MIT Cheetah) or opposed (Anymal) configuration.
        valid_filter: Optional callable to filter valid parameter combinations.
    
    Example:
        >>> builder = QuadrupedBuilder(
        ...     base_length_range=(0.5, 1.0),
        ...     leg_length_range=(0.4, 0.8)
        ... )
        >>> param = builder.sample_params(seed=42)
        >>> spec = builder.generate_mjspec(param)
    """
        
    def __init__(
        self,
        base_length_range: tuple[float, float] = (0.5, 1.0),
        base_width_range: tuple[float, float] = (0.3, 0.4),
        base_height_range: tuple[float, float] = (0.15, 0.25),
        leg_length_range: tuple[float, float] = (0.4, 0.8),
        calf_length_ratio: tuple[float, float] = (0.9, 1.0),
        parallel_abduction: float = 0.5,
        valid_filter: Callable[[QuadrupedParam], bool] = lambda _: True,
    ):
        super().__init__()
        self.base_length_range = base_length_range
        self.base_width_range = base_width_range
        self.base_height_range = base_height_range
        self.leg_length_range = leg_length_range
        self.calf_length_ratio = calf_length_ratio
        self.parallel_abduction = parallel_abduction
        self.valid_filter = valid_filter
    
    def sample_params(self, seed: int=-1):
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)
        
        for _ in range(10):
            base_length = random.uniform(*self.base_length_range)
            base_width = random.uniform(*self.base_width_range)
            base_height = random.uniform(*self.base_height_range)
            thigh_length = base_length * random.uniform(*self.leg_length_range)
            calf_length = thigh_length * random.uniform(*self.calf_length_ratio)
            thigh_radius = random.uniform(0.03, 0.05)
            parallel_abduction = random.random() < self.parallel_abduction
            param = QuadrupedParam(
                base_length=base_length,
                base_width=base_width,
                base_height=base_height,
                thigh_length=thigh_length,
                calf_length=calf_length,
                thigh_radius=thigh_radius,
                parallel_abduction=parallel_abduction,
            )
            if self.valid_filter(param):
                break
        else:
            raise ValueError("Failed to sample valid parameters")
        return param
    
    def generate_mjspec(self, param: QuadrupedParam) -> mujoco.MjSpec:
        thigh_radius = param.thigh_radius
        calf_radius = param.thigh_radius * 0.8
        foot_radius = param.thigh_radius * 0.9

        spec = mujoco.MjSpec()
        base_body = spec.worldbody.add_body()
        base_body.name = "base"
        base_body.mass = 1.0
        base_body.inertia = [1.0, 1.0, 1.0]
        trunk_geom = base_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
        trunk_geom.size = [param.base_length/2, param.base_width/2, param.base_height/2]

        for name, (x, y) in [
            ("FL", (1, -1)),
            ("FR", (1, 1)),
            ("RL", (-1, -1)),
            ("RR", (-1, 1))
        ]:
            hip_body = base_body.add_body(name=f"{name}_hip")
            hip_body.pos = [0.5 * param.base_length * x, 0.5 * param.base_width * y, 0]
            hip_body.mass = 1.0
            hip_body.inertia = [1.0, 1.0, 1.0]
            joint = hip_body.add_joint(name=f"{name}_hip_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0])
            joint.range = [-np.pi * 0.6, +np.pi * 0.6]
            add_capsule_geom_(hip_body, radius=thigh_radius * 1.8, fromto=[-0.1, 0, 0, 0.1, 0, 0])

            thigh_body = hip_body.add_body(name=f"{name}_thigh")
            thigh_body.pos = [0.1 * x, 0.1 * y, 0]
            thigh_body.mass = 1.0
            thigh_body.inertia = [1.0, 1.0, 1.0]
            joint = thigh_body.add_joint(name=f"{name}_thigh_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
            joint.range = [-np.pi * 0.8, np.pi * 0.8]
            add_capsule_geom_(thigh_body, radius=thigh_radius, fromto=[0, 0, 0, 0, 0, -param.thigh_length])

            calf_body = thigh_body.add_body(name=f"{name}_calf")
            calf_body.pos = [0, 0, -param.thigh_length]
            calf_body.mass = 1.0
            calf_body.inertia = [1.0, 1.0, 1.0]
            joint = calf_body.add_joint(name=f"{name}_calf_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
            joint.range = [-np.pi, np.pi]
            add_capsule_geom_(calf_body, radius=calf_radius, fromto=[0, 0, 0, 0, 0, -param.calf_length])

            feet_body = calf_body.add_body(name=f"{name}_foot")
            feet_body.pos = [0, 0, -param.calf_length]
            feet_body.mass = 1.0
            feet_body.inertia = [1.0, 1.0, 1.0]
            feet_geom = feet_body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE)
            feet_geom.size = [foot_radius, 0., 0.]

        return spec
    
    @staticmethod
    def _compute_standing_pose(param: QuadrupedParam) -> tuple[float, float, float]:
        """Compute default thigh_angle, calf_angle, and standing height for a given morphology.
        
        Strategy: choose calf_angle so the calf is roughly vertical,
        then set thigh_angle to get a natural stance.
        
        The leg kinematic chain (in the sagittal plane, angles relative to vertical):
          - thigh hangs from hip, rotated by thigh_angle from vertical (positive = forward)
          - calf hangs from knee, rotated by calf_angle from thigh direction (negative = bend back)
        
        Foot z below hip = thigh_length * cos(thigh_angle) + calf_length * cos(thigh_angle + calf_angle)
        
        We want a ratio of leg_height / total_leg_length ~ 0.7 for a natural crouch.
        
        Returns:
            (front_thigh_angle, front_calf_angle, standing_height)
        """
        total_leg = param.thigh_length + param.calf_length
        foot_radius = param.thigh_radius * 0.9
        
        # Target: leg vertical reach ~ 70% of full extension for a stable crouch
        target_reach = total_leg * 0.7
        
        # Use a simple geometric approach:
        # thigh_angle ≈ acos((target_reach^2 + thigh^2 - calf^2) / (2 * target_reach * thigh))
        # But simpler: pick thigh_angle based on leg ratio, then solve calf_angle
        # 
        # For typical quadrupeds:
        #   thigh_angle ~ 0.6-0.8 rad (forward lean)
        #   calf_angle ~ -(thigh_angle + extra) to keep foot roughly under hip
        
        # Scale thigh angle with leg proportion - longer legs need less bend
        thigh_angle = 0.7  # ~40 degrees forward from vertical
        
        # calf_angle chosen so foot is roughly under the hip (small forward offset is ok)
        # foot_x = thigh * sin(thigh_angle) + calf * sin(thigh_angle + calf_angle) ≈ 0
        # => sin(thigh_angle + calf_angle) ≈ -thigh/calf * sin(thigh_angle)
        sin_knee = -param.thigh_length / param.calf_length * np.sin(thigh_angle)
        sin_knee = np.clip(sin_knee, -1.0, 1.0)
        calf_angle = np.arcsin(sin_knee) - thigh_angle
        # Clamp to reasonable range
        calf_angle = np.clip(calf_angle, -2.5, -0.5)
        
        # Compute actual standing height (vertical distance from hip to foot)
        leg_height = (
            param.thigh_length * np.cos(thigh_angle)
            + param.calf_length * np.cos(thigh_angle + calf_angle)
        )
        standing_height = leg_height + foot_radius + param.base_height / 2 + 0.05
        
        return float(thigh_angle), float(calf_angle), float(standing_height)

    @classmethod
    def modify_articulation(cls, articulation: Articulation):
        """Modify the articulation's default joint positions, joint position limits,
        and default root state based on each robot's morphology parameters.

        This computes per-robot standing poses from the sampled params, overriding
        whatever was set in the env cfg's init_state.

        Args:
            articulation: The articulation to modify.
        """
        builder: QuadrupedBuilder = cls.get_instance()
        device = articulation.device
        num_envs = articulation.num_instances
        
        parallel_abduction = torch.tensor(
            [p.parallel_abduction for p in builder.params],
            device=device,
        )
        
        hip_joint_ids = articulation.find_joints(".*_hip_joint")[0]
        front_thigh_joint_ids = articulation.find_joints("F[L,R]_thigh_joint")[0]
        front_calf_joint_ids = articulation.find_joints("F[L,R]_calf_joint")[0]
        rear_thigh_joint_ids = articulation.find_joints("R[L,R]_thigh_joint")[0]
        rear_calf_joint_ids = articulation.find_joints("R[L,R]_calf_joint")[0]

        # --- Compute per-robot default joint positions and heights ---
        default_joint_pos = torch.zeros_like(articulation.data.default_joint_pos)
        heights = []
        
        for i, param in enumerate(builder.params):
            front_thigh, front_calf, height = cls._compute_standing_pose(param)
            heights.append(height)
            
            # Hip joints: 0
            default_joint_pos[i, hip_joint_ids] = 0.0
            # Front legs
            default_joint_pos[i, front_thigh_joint_ids] = front_thigh
            default_joint_pos[i, front_calf_joint_ids] = front_calf
            # Rear legs: depend on parallel_abduction
            if param.parallel_abduction:
                default_joint_pos[i, rear_thigh_joint_ids] = front_thigh
                default_joint_pos[i, rear_calf_joint_ids] = front_calf
            else:
                default_joint_pos[i, rear_thigh_joint_ids] = -front_thigh
                default_joint_pos[i, rear_calf_joint_ids] = -front_calf
        
        articulation.data.default_joint_pos[:] = default_joint_pos
        
        # Write joint positions to simulation so they take effect
        articulation.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos))

        # --- Set joint position limits ---
        joint_pos_limits = articulation.data.joint_pos_limits.clone()
        joint_pos_limits[:, front_calf_joint_ids] = torch.tensor([-torch.pi, 0.0], device=device)
        joint_pos_limits[:, rear_calf_joint_ids] = torch.where(
            parallel_abduction.reshape(num_envs, 1, 1),
            torch.tensor([-torch.pi, 0.0], device=device),
            torch.tensor([0.0, torch.pi], device=device),
        )
        articulation.data.joint_pos_limits[:] = joint_pos_limits
        articulation.write_joint_position_limit_to_sim(joint_pos_limits)
        
        # --- Set per-robot initial root height and position ---
        default_root_state = articulation.data.default_root_state.clone()
        for i, height in enumerate(heights):
            default_root_state[i, 2] = height
        articulation.data.default_root_state[:] = default_root_state
        articulation.write_root_state_to_sim(default_root_state)
        
        print(f"[INFO] Procedural quadruped standing poses computed:")
        print(f"       Heights: min={min(heights):.3f}m, max={max(heights):.3f}m, mean={sum(heights)/len(heights):.3f}m")
        print(f"       Example env0: thigh={default_joint_pos[0, front_thigh_joint_ids[0]]:.3f}, "
              f"calf={default_joint_pos[0, front_calf_joint_ids[0]]:.3f}, h={heights[0]:.3f}m")


if __name__ == "__main__":
    builder = QuadrupedBuilder()
    builder_instance = QuadrupedBuilder.get_instance()
    assert builder_instance is builder
    # builder = QuadrupedBuilder()
    param = builder.sample_params()
    print(param)

