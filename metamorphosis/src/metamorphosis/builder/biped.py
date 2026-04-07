import mujoco
import numpy as np
import random

from typing import NamedTuple, Callable
from metamorphosis.builder.base import BuilderBase
from metamorphosis.utils.mjs_utils import add_capsule_geom_


class BipedParam(NamedTuple):
    """Parameters for a bipedal robot with box pelvis and standard humanoid hip structure."""
    # Torso (main body - box)
    torso_link_length: float    
    torso_link_width: float      
    torso_link_height: float    
    torso_link_mass: float       
    # Waist connection
    pelvis_height: float      # waist connection height
    pelvis_radius: float      # waist radius (small cylinder)
    
    # Hip positioning and dimensions
    hip_spacing: float       # left-right hip spacing
    
    # Hip_pitch_link parameters (first hip segment)
    hip_pitch_link_length: float       # hip pitch link length
    hip_pitch_link_radius: float       # hip pitch link radius
    hip_pitch_link_mass: float         # hip pitch link mass
    
    # Hip_roll_link parameters (second hip segment)
    hip_roll_link_length: float       # hip roll link length
    hip_roll_link_radius: float       # hip roll link radius
    hip_roll_link_mass: float         # hip roll link mass

    # Hip pitch roll initialization
    hip_pitch_link_initroll: float    # initial outward roll angle of hip pitch link around X-axis
    
    # hip_yaw_link parameters  
    hip_yaw_link_length: float      # thigh length
    hip_yaw_link_radius: float      # thigh radius
    hip_yaw_link_mass: float        # thigh mass
    
    # knee_link parameters
    knee_link_length: float       # shin length
    knee_link_radius: float       # shin radius
    knee_link_mass: float         # shin mass
    
    # ankle_pitch_link parameters
    ankle_pitch_link_length: float       # ankle pitch link length
    ankle_pitch_link_radius: float       # ankle pitch link radius
    ankle_pitch_link_mass: float         # ankle pitch link mass
    
    # ankle_roll_link parameters
    ankle_roll_link_length: float       # foot length
    ankle_roll_link_width: float        # foot width
    ankle_roll_link_height: float       # foot height
    ankle_roll_link_mass: float         # foot mass


class BipedBuilder(BuilderBase):
    """
    Builder for generating procedurally parameterized bipedal robots (legs only).
    
    This class creates bipedal robot models with configurable body dimensions
    and leg proportions. It samples random parameters within specified ranges
    and generates MuJoCo specifications for the robot model.
    
    The biped uses a standard humanoid structure:
    - A pelvis (base body) with configurable dimensions
    - Two legs, each with:
      - Waist connection geometry (cylinder)
      - Hip joint 1 (pitch) -> Hip1 body (capsule)
      - Hip joint 2 (roll) -> Hip2 body (capsule) 
      - Hip joint 3 (yaw) -> Thigh body (capsule)
      - Knee joint (pitch) -> Shin/Calf body (capsule)
      - Ankle pitch joint (virtual body)
      - Ankle roll joint -> Foot body (box)
    
    This design follows: pelvis geo -> hip_pitch_joint -> hip_pitch_link -> hip_roll_joint -> hip_roll_link -> 
    hip_yaw_joint -> hip_yaw_link -> knee joint -> knee_link -> ankle_pitch_joint -> ankle_pitch_link -> ankle_roll_joint -> ankle_roll_link (foot)
    
    Attributes:
        torso_link_length_range/torso_link_width_range/torso_link_height_range: Box torso dimensions.
        pelvis_height_range: Range for waist cylinder height (also used for offset below torso).
        hip_spacing_range: Left/right hip separation.
        hip_pitch_link_length_range/hip_pitch_link_radius_range: First hip segment size ranges.
        hip_roll_link_length_range/hip_roll_link_radius_range: Second hip segment size ranges.
        hip_pitch_link_initroll_range: Range for initial outward roll of hip pitch bodies.
        leg_length_range: Range for total leg length (thigh + shin).
        shin_ratio_range: Ratio of shin to thigh length.
        valid_filter: Optional callable to filter valid parameter combinations.
    
    Example:
        >>> builder = BipedBuilder(
        ...     pelvis_height_range=(0.10, 0.15),
        ...     leg_length_range=(0.6, 1.0)
        ... )
        >>> param = builder.sample_params(seed=42)
        >>> spec = builder.generate_mjspec(param)
    """
    
    def __init__(
        self,
        torso_link_length_range: tuple[float, float] = (0.10, 0.16),
        torso_link_width_range: tuple[float, float] = (0.18, 0.26),
        torso_link_height_range: tuple[float, float] = (0.5, 2.0),
        pelvis_height_range: tuple[float, float] = (0.05, 0.08),
        hip_spacing_range: tuple[float, float] = (0.16, 0.24),
        hip_pitch_link_length_range: tuple[float, float] = (0.03, 0.06),  # Hip1 length range
        hip_pitch_link_radius_range: tuple[float, float] = (0.02, 0.04),  # Hip1 radius range
        hip_roll_link_length_range: tuple[float, float] = (0.03, 0.06),  # Hip2 length range
        hip_roll_link_radius_range: tuple[float, float] = (0.02, 0.04),  # Hip2 radius range
        hip_pitch_link_initroll_range: tuple[float, float] = (0.0, np.pi / 2),
        leg_length_range: tuple[float, float] = (0.5, 0.7),
        shin_ratio_range: tuple[float, float] = (0.85, 1.15),
        valid_filter: Callable[[BipedParam], bool] = lambda _: True,
    ):
        super().__init__()
        self.torso_link_length_range = torso_link_length_range
        self.torso_link_width_range = torso_link_width_range
        self.torso_link_height_range = torso_link_height_range
        self.pelvis_height_range = pelvis_height_range
        self.hip_spacing_range = hip_spacing_range
        self.hip_pitch_link_length_range = hip_pitch_link_length_range
        self.hip_pitch_link_radius_range = hip_pitch_link_radius_range
        self.hip_roll_link_length_range = hip_roll_link_length_range
        self.hip_roll_link_radius_range = hip_roll_link_radius_range
        self.hip_pitch_link_initroll_range = hip_pitch_link_initroll_range
        self.leg_length_range = leg_length_range
        self.shin_ratio_range = shin_ratio_range
        self.valid_filter = valid_filter
    
    def sample_params(self, seed: int = -1) -> BipedParam:
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)

        for _ in range(10):
            # Torso (box)
            torso_link_length = random.uniform(*self.torso_link_length_range)
            torso_link_width = random.uniform(*self.torso_link_width_range)
            torso_link_height = random.uniform(*self.torso_link_height_range)
            torso_link_mass = random.uniform(3.0, 5.0)

            # Waist connection
            pelvis_height = random.uniform(*self.pelvis_height_range)
            pelvis_radius = min(torso_link_width, torso_link_length) * random.uniform(0.2, 0.4)
            # Hip spacing and hip segment dimensions
            hip_spacing = random.uniform(*self.hip_spacing_range)

            # Hip pitch segment (first hip)
            hip_pitch_link_length = random.uniform(*self.hip_pitch_link_length_range)
            hip_pitch_link_radius = random.uniform(*self.hip_pitch_link_radius_range)
            hip_pitch_link_mass = random.uniform(0.4, 0.8)

            # Hip roll segment (second hip)
            hip_roll_link_length = random.uniform(*self.hip_roll_link_length_range)
            hip_roll_link_radius = random.uniform(*self.hip_roll_link_radius_range)
            hip_roll_link_mass = random.uniform(0.4, 0.8)

            # Hip pitch initial roll (0 to 90 degrees)
            hip_pitch_link_initroll = random.uniform(*self.hip_pitch_link_initroll_range)

            # Leg dimensions: split total leg into thigh (hip_yaw_link) and shin (knee_link)
            total_leg_length = random.uniform(*self.leg_length_range)
            shin_ratio = random.uniform(*self.shin_ratio_range)
            hip_yaw_link_length = total_leg_length / (1 + shin_ratio)
            knee_link_length = hip_yaw_link_length * shin_ratio

            # Leg masses and radii (proportional to length)
            hip_yaw_link_radius = random.uniform(0.025, 0.040)
            hip_yaw_link_mass = hip_yaw_link_length * random.uniform(1.5, 2.2)

            knee_link_radius = hip_yaw_link_radius * random.uniform(0.75, 0.95)
            knee_link_mass = knee_link_length * random.uniform(1.2, 1.8)

            # Foot parameters (ankle roll link)
            ankle_roll_link_length = random.uniform(0.18, 0.25)
            ankle_roll_link_width = random.uniform(0.06, 0.10)
            ankle_roll_link_height = random.uniform(0.02, 0.03)
            ankle_roll_link_mass = random.uniform(0.2, 0.5)

            # Ankle pitch link (virtual body) derived from foot size
            ankle_pitch_link_radius = min(ankle_roll_link_width, ankle_roll_link_height) / 2.0
            ankle_pitch_link_length = ankle_pitch_link_radius * 2.0
            ankle_pitch_link_mass = 0.3

            param = BipedParam(
                torso_link_length=torso_link_length,
                torso_link_width=torso_link_width,
                torso_link_height=torso_link_height,
                torso_link_mass=torso_link_mass,
                pelvis_height=pelvis_height,
                pelvis_radius=pelvis_radius,
                hip_spacing=hip_spacing,
                hip_pitch_link_length=hip_pitch_link_length,
                hip_pitch_link_radius=hip_pitch_link_radius,
                hip_pitch_link_mass=hip_pitch_link_mass,
                hip_roll_link_length=hip_roll_link_length,
                hip_roll_link_radius=hip_roll_link_radius,
                hip_roll_link_mass=hip_roll_link_mass,
                hip_pitch_link_initroll=hip_pitch_link_initroll,
                hip_yaw_link_length=hip_yaw_link_length,
                hip_yaw_link_radius=hip_yaw_link_radius,
                hip_yaw_link_mass=hip_yaw_link_mass,
                knee_link_length=knee_link_length,
                knee_link_radius=knee_link_radius,
                knee_link_mass=knee_link_mass,
                ankle_pitch_link_length=ankle_pitch_link_length,
                ankle_pitch_link_radius=ankle_pitch_link_radius,
                ankle_pitch_link_mass=ankle_pitch_link_mass,
                ankle_roll_link_length=ankle_roll_link_length,
                ankle_roll_link_width=ankle_roll_link_width,
                ankle_roll_link_height=ankle_roll_link_height,
                ankle_roll_link_mass=ankle_roll_link_mass,
            )

            if self.valid_filter(param):
                break
        else:
            raise ValueError("Failed to sample valid parameters")

        return param
    
    def generate_mjspec(self, param: BipedParam) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        
        # ============ Torso ============
        # Add torso body
        torso_body = spec.worldbody.add_body(name="torso_link")
        torso_body.mass = param.torso_link_mass
        # Torso inertia
        torso_link_ixx = param.torso_link_mass * (param.torso_link_width**2 + param.torso_link_height**2) / 12
        torso_link_iyy = param.torso_link_mass * (param.torso_link_length**2 + param.torso_link_height**2) / 12
        torso_link_izz = param.torso_link_mass * (param.torso_link_length**2 + param.torso_link_width**2) / 12
        torso_body.inertia = [torso_link_ixx, torso_link_iyy, torso_link_izz]
        # Torso geometry
        torso_geom = torso_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
        torso_geom.size = [param.torso_link_length/2, param.torso_link_width/2, param.torso_link_height/2]
        torso_geom.rgba = [0.75, 0.75, 0.75, 1.0]
        
        # ============ Pelvis ============
        # Add pelvis
        pelvis = torso_body.add_body(name="pelvis")
        pelvis.pos = [0, 0, -(param.torso_link_height/2 + param.pelvis_height/2)]
        pelvis.mass = 0.5
        # Pelvis inertia
        pelvis.inertia = [0.01, 0.01, 0.01]
        # Pelvis geometry
        pelvis_link = pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER)
        pelvis_link.size = [param.pelvis_radius, param.pelvis_height/2, 0]
        pelvis_link.rgba = [0.65, 0.65, 0.85, 1.0]
        
        # ============ LEFT AND RIGHT LEGS (6 DOF each) ============
        for side, y_sign in [("left", 1), ("right", -1)]:
            
            # --- hip_pitch_joint ---
            # Add hip_pitch_joint body
            hip_pitch_joint = pelvis.add_body(name=f"{side}_hip_pitch_joint")
            hip_pitch_joint.pos = [0, y_sign * param.hip_spacing/2, -(param.pelvis_height/2)]
            hip_pitch_roll = param.hip_pitch_link_initroll if side == "left" else -param.hip_pitch_link_initroll
            hip_pitch_joint.quat = [
                np.cos(hip_pitch_roll / 2.0),
                np.sin(hip_pitch_roll / 2.0),
                0.0,
                0.0,
            ]
            hip_pitch_joint.mass = param.hip_pitch_link_mass
            # Add axis and rotation center for hip_pitch_joint
            pitch_joint = hip_pitch_joint.add_joint(
                name=f"{side}_hip_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            pitch_joint.axis = [1, 0, 0]  # X-axis (roll)
            pitch_joint.range = [-2.5307, 2.8798]
            pitch_joint.armature = 0.01017752004
            
            # --- hip_pitch_link ---
            # Add hip_pitch_link inertia
            hip_pitch_link_ixx = param.hip_pitch_link_mass * (3 * param.hip_pitch_link_radius**2 + param.hip_pitch_link_length**2) / 12
            hip_pitch_link_iyy = param.hip_pitch_link_mass * param.hip_pitch_link_radius**2 / 2
            hip_pitch_link_izz = hip_pitch_link_ixx
            hip_pitch_joint.inertia = [hip_pitch_link_ixx, hip_pitch_link_iyy, hip_pitch_link_izz]
            # Add hip_pitch_link geometry
            hip_pitch_link = hip_pitch_joint.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_pitch_link.name = f"{side}_hip_pitch_link"
            hip_pitch_link.size = [param.hip_pitch_link_radius, 0, 0]
            hip_pitch_link.fromto = [0, 0, 0, 0, 0, -param.hip_pitch_link_length]
            hip_pitch_link.rgba = [0.85, 0.45, 0.45, 1.0]
            
            # --- hip_roll_joint ---
            # Add hip_roll_joint body
            hip_roll_joint = hip_pitch_joint.add_body(name=f"{side}_hip_roll_joint")
            hip_roll_joint.pos = [0, 0, -(param.hip_pitch_link_length + param.hip_pitch_link_radius + param.hip_roll_link_radius)]  # Tangent to hip_pitch_link
            hip_roll_joint.quat = [
                np.cos(-hip_pitch_roll / 2.0),
                np.sin(-hip_pitch_roll / 2.0),
                0.0,
                0.0,
            ]
            hip_roll_joint.mass = param.hip_roll_link_mass
            # Add axis and rotation center for hip_roll_joint
            roll_joint = hip_roll_joint.add_joint(
                name=f"{side}_hip_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            # Different ranges for left/right legs
            if side == "left":
                roll_joint.range = [-0.5236, 2.9671]
            else:
                roll_joint.range = [-2.9671, 0.5236]
            roll_joint.armature = 0.025101925
            
            # --- hip_roll_link ---
            # Add hip_roll_link inertia
            hip_roll_link_ixx = param.hip_roll_link_mass * (3 * param.hip_roll_link_radius**2 + param.hip_roll_link_length**2) / 12
            hip_roll_link_iyy = param.hip_roll_link_mass * param.hip_roll_link_radius**2 / 2
            hip_roll_link_izz = hip_roll_link_ixx
            hip_roll_joint.inertia = [hip_roll_link_ixx, hip_roll_link_iyy, hip_roll_link_izz]
            # Add hip_roll_link geometry
            hip_roll_link = hip_roll_joint.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_roll_link.name = f"{side}_hip_roll_link"
            hip_roll_link.size = [param.hip_roll_link_radius, 0, 0]
            hip_roll_link.fromto = [0, 0, 0, 0, 0, -param.hip_roll_link_length]
            hip_roll_link.rgba = [0.45, 0.75, 0.45, 1.0]
            
            # --- hip_yaw_joint ---
            # Add hip_yaw_joint body
            hip_yaw_joint = hip_roll_joint.add_body(name=f"{side}_hip_yaw_joint")
            hip_yaw_joint.pos = [0, 0, -(param.hip_roll_link_length + param.hip_roll_link_radius + param.hip_yaw_link_radius)]  # Tangent to hip_roll_link
            hip_yaw_joint.mass = param.hip_yaw_link_mass
            hip_yaw_joint.quat = [1.0, 0.0, 0.0, 0.0]
            # Add axis and rotation center for hip_yaw_joint
            yaw_joint = hip_yaw_joint.add_joint(
                name=f"{side}_hip_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            yaw_joint.axis = [0, 0, 1]  # Z-axis (yaw)
            yaw_joint.range = [-2.7576, 2.7576]
            yaw_joint.armature = 0.01017752004
            
            # --- hip_yaw_link ---
            # Add hip_yaw_link inertia
            hip_yaw_link_ixx = param.hip_yaw_link_mass * (3 * param.hip_yaw_link_radius**2 + param.hip_yaw_link_length**2) / 12
            hip_yaw_link_iyy = param.hip_yaw_link_mass * param.hip_yaw_link_radius**2 / 2
            hip_yaw_link_izz = hip_yaw_link_ixx
            hip_yaw_joint.inertia = [hip_yaw_link_ixx, hip_yaw_link_iyy, hip_yaw_link_izz]
            # Add hip_yaw_link geometry
            hip_yaw_geom = hip_yaw_joint.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hip_yaw_geom.name = f"{side}_hip_yaw_link"
            hip_yaw_geom.size = [param.hip_yaw_link_radius, 0, 0]
            hip_yaw_geom.fromto = [0, 0, 0, 0, 0, -param.hip_yaw_link_length]
            hip_yaw_geom.rgba = [0.45, 0.45, 0.85, 1.0]
            
            # --- knee_joint ---
            # Add knee_joint body
            knee_joint = hip_yaw_joint.add_body(name=f"{side}_knee_joint")
            knee_joint.pos = [0, 0, -(param.hip_yaw_link_length + param.hip_yaw_link_radius + param.knee_link_radius)] 
            knee_joint.mass = param.knee_link_mass
            knee_joint.quat = [1.0, 0.0, 0.0, 0.0]
            # Add axis and rotation center for knee_joint
            pitch_joint = knee_joint.add_joint(
                name=f"{side}_knee_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            pitch_joint.range = [-0.087267, 2.8798]
            pitch_joint.armature = 0.025101925
            
            # --- knee_link ---
            # Shin inertia
            knee_link_ixx = param.knee_link_mass * (3 * param.knee_link_radius**2 + param.knee_link_length**2) / 12
            knee_link_iyy = param.knee_link_mass * param.knee_link_radius**2 / 2
            knee_link_izz = knee_link_ixx
            knee_joint.inertia = [knee_link_ixx, knee_link_iyy, knee_link_izz]
            
            # Knee geometry - capsule
            knee_link_geom = knee_joint.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            knee_link_geom.name = f"{side}_knee_link"
            knee_link_geom.size = [param.knee_link_radius, 0, 0]
            knee_link_geom.fromto = [0, 0, 0, 0, 0, -param.knee_link_length]
            knee_link_geom.rgba = [0.85, 0.75, 0.35, 1.0]
            
            # --- ANKLE JOINT 1: PITCH ---
            ankle_pitch_link = knee_joint.add_body(name=f"{side}_ankle_pitch_link")
            ankle_pitch_link.pos = [0, 0, -(param.knee_link_length + param.knee_link_radius + param.ankle_pitch_link_radius)]  # Tangent to knee_link
            ankle_pitch_link.mass = param.ankle_pitch_link_mass
            
            ankle_pitch_joint = ankle_pitch_link.add_joint(
                name=f"{side}_ankle_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch)
            ankle_pitch_joint.range = [-0.87267, 0.5236]
            ankle_pitch_joint.armature = 0.00721945
            
            ankle_pitch_link.inertia = [0.005, 0.005, 0.005]

            ankle_pitch_geom = ankle_pitch_link.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE)
            ankle_pitch_geom.size = [param.ankle_pitch_link_radius, 0, 0]
            ankle_pitch_geom.rgba = [0.55, 0.85, 0.85, 1.0]
            
            # --- ANKLE JOINT 2: ROLL (FOOT) ---
            ankle_roll_link = ankle_pitch_link.add_body(name=f"{side}_ankle_roll_link")
            ankle_roll_link.pos = [0, 0, -(param.ankle_pitch_link_radius + param.ankle_roll_link_height/2)]  # Tangent to ankle_pitch_link
            ankle_roll_link.mass = param.ankle_roll_link_mass
            
            ankle_roll_joint = ankle_roll_link.add_joint(
                name=f"{side}_ankle_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE
            )
            ankle_roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            ankle_roll_joint.range = [-0.2618, 0.2618]
            ankle_roll_joint.armature = 0.00721945
            
            # Foot inertia
            foot_ixx = param.ankle_roll_link_mass * (param.ankle_roll_link_width**2 + param.ankle_roll_link_height**2) / 12
            foot_iyy = param.ankle_roll_link_mass * (param.ankle_roll_link_length**2 + param.ankle_roll_link_height**2) / 12
            foot_izz = param.ankle_roll_link_mass * (param.ankle_roll_link_length**2 + param.ankle_roll_link_width**2) / 12
            ankle_roll_link.inertia = [foot_ixx, foot_iyy, foot_izz]
            
            # Foot geometry - box extending forward
            foot_geom = ankle_roll_link.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
            foot_geom.size = [param.ankle_roll_link_length/2, param.ankle_roll_link_width/2, param.ankle_roll_link_height/2]
            foot_geom.pos = [param.ankle_roll_link_length/4, 0, -param.ankle_roll_link_height/2]
            foot_geom.rgba = [0.85, 0.55, 0.85, 1.0]
        
        return spec


if __name__ == "__main__":
    builder = BipedBuilder()
    param = builder.sample_params(seed=0)
    print(param)
    spec = builder.generate_mjspec(param)
    print("Generated biped MjSpec successfully!")
