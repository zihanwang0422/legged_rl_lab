import mujoco
import numpy as np
import random

from typing import NamedTuple, Callable
from metamorphosis.builder.base import BuilderBase
from metamorphosis.utils.mjs_utils import add_capsule_geom_


class QuadWheelParam(NamedTuple):
    base_length: float
    base_width: float
    base_height: float
    thigh_length: float
    calf_length: float
    thigh_radius: float
    wheel_radius: float
    wheel_width: float


class QuadWheelBuilder(BuilderBase):
    """
    Builder for generating procedurally parameterized four-wheeled robots.
    
    This class creates four-wheeled robot models with configurable body dimensions,
    leg proportions, and wheel parameters. It samples random parameters within 
    specified ranges and generates MuJoCo specifications for the robot model.
    
    The quad-wheel robot consists of:
    - A base body (trunk) with configurable length, width, and height
    - Four legs, each with:
      - A hip joint (abduction/adduction)
      - A thigh segment with a knee joint
      - A calf segment
      - A wheel (cylindrical shape with continuous rotation joint)
    
    Attributes:
        base_length_range: Tuple of (min, max) for base body length in meters.
        base_width_range: Tuple of (min, max) for base body width in meters.
        base_height_range: Tuple of (min, max) for base body height in meters.
        leg_length_range: Tuple of (min, max) for leg length as a ratio of base_length.
        calf_length_ratio: Tuple of (min, max) for calf length as a ratio of thigh_length.
        wheel_radius_range: Tuple of (min, max) for wheel radius in meters.
        wheel_width_range: Tuple of (min, max) for wheel width in meters.
        valid_filter: Optional callable to filter valid parameter combinations.
    
    Example:
        >>> builder = QuadWheelBuilder(
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
        wheel_radius_range: tuple[float, float] = (0.08, 0.15),
        wheel_width_range: tuple[float, float] = (0.03, 0.06),
        valid_filter: Callable[[QuadWheelParam], bool] = lambda _: True,
    ):
        super().__init__()
        self.base_length_range = base_length_range
        self.base_width_range = base_width_range
        self.base_height_range = base_height_range
        self.leg_length_range = leg_length_range
        self.calf_length_ratio = calf_length_ratio
        self.wheel_radius_range = wheel_radius_range
        self.wheel_width_range = wheel_width_range
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
            wheel_radius = random.uniform(*self.wheel_radius_range)
            wheel_width = random.uniform(*self.wheel_width_range)
            param = QuadWheelParam(
                base_length=base_length,
                base_width=base_width,
                base_height=base_height,
                thigh_length=thigh_length,
                calf_length=calf_length,
                thigh_radius=thigh_radius,
                wheel_radius=wheel_radius,
                wheel_width=wheel_width,
            )
            if self.valid_filter(param):
                break
        else:
            raise ValueError("Failed to sample valid parameters")
        return param
    
    def generate_mjspec(self, param: QuadWheelParam) -> mujoco.MjSpec:
        thigh_radius = param.thigh_radius
        calf_radius = param.thigh_radius * 0.8

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
            joint.range = [-np.pi, 0]
            add_capsule_geom_(calf_body, radius=calf_radius, fromto=[0, 0, 0, 0, 0, -param.calf_length])

            # Wheel body with continuous rotation joint
            # Position wheel at the end of calf, offset to the side (Y direction)
            wheel_body = calf_body.add_body(name=f"{name}_wheel")
            wheel_body.pos = [0, y * (param.wheel_radius + calf_radius), -param.calf_length]  # Offset to outer side
            wheel_body.mass = 0.5
            wheel_body.inertia = [0.1, 0.1, 0.1]
            
            # Continuous rotation joint for wheel (rotation around Y-axis, like car wheels)
            wheel_joint = wheel_body.add_joint(
                name=f"{name}_wheel_joint", 
                type=mujoco.mjtJoint.mjJNT_HINGE, 
                axis=[0, 1, 0]  # Rotation around Y-axis (forward/backward rolling)
            )
            # No range specified = unlimited rotation
            
            # Cylinder geometry for wheel (horizontal, axis along Y)
            wheel_geom = wheel_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER)
            wheel_geom.size = [param.wheel_radius, param.wheel_width / 2, 0.]
            # Rotate cylinder 90 degrees around X-axis to align with Y-axis (horizontal)
            wheel_geom.quat = [0.7071068, 0.7071068, 0, 0]  # 90 degree rotation around X-axis

        return spec


if __name__ == "__main__":
    builder = QuadWheelBuilder()
    builder_instance = QuadWheelBuilder.get_instance()
    assert builder_instance is builder
    param = builder.sample_params()
    print(param)
