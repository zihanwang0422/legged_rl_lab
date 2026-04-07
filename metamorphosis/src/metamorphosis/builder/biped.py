from __future__ import annotations

import mujoco
import numpy as np
import random
import torch

from typing import NamedTuple, Callable, TYPE_CHECKING
from metamorphosis.builder.base import BuilderBase
from metamorphosis.utils.mjs_utils import add_capsule_geom_

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


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

    # Head (sphere on top of torso)
    head_radius: float
    head_mass: float

    # Arms (symmetric per side: shoulder_pitch/roll/yaw + elbow + wrist_roll/pitch/yaw)
    upper_arm_length: float    # shoulder_yaw_link / humerus
    upper_arm_radius: float
    upper_arm_mass: float
    forearm_length: float      # elbow_link / forearm
    forearm_radius: float
    forearm_mass: float


class BipedBuilder(BuilderBase):
    """
    Builder for generating procedurally parameterized bipedal robots with arms and head (G1-style).

    Body structure (G1 joint/link naming convention):
        torso_link (box)
        ├── head_link (sphere, fixed atop torso)
        ├── pelvis (cylinder, below torso)
        │   ├── {side}_hip_pitch_link  → hip_pitch_joint  (Y)
        │   │   └── {side}_hip_roll_link  → hip_roll_joint  (X)
        │   │       └── {side}_hip_yaw_link  → hip_yaw_joint  (Z, thigh)
        │   │           └── {side}_knee_link  → knee_joint  (Y, shin)
        │   │               └── {side}_ankle_pitch_link → ankle_pitch_joint (Y)
        │   │                   └── {side}_ankle_roll_link → ankle_roll_joint (X, foot)
        ├── {side}_shoulder_pitch_link → shoulder_pitch_joint (Y)
        │   └── {side}_shoulder_roll_link → shoulder_roll_joint (X)
        │       └── {side}_shoulder_yaw_link → shoulder_yaw_joint (Z, upper arm)
        │           └── {side}_elbow_link → elbow_joint (Y, forearm)
        │               └── {side}_wrist_roll_link → wrist_roll_joint (X)
        │                   └── {side}_wrist_pitch_link → wrist_pitch_joint (Y)
        │                       └── {side}_wrist_yaw_link → wrist_yaw_joint (Z)

    Attributes:
        torso_link_length/width/height_range: Box torso dimensions.
        pelvis_height_range: Waist cylinder height.
        hip_spacing_range: Left-right hip separation.
        hip_pitch/roll_link_length/radius_range: Hip segment sizes.
        hip_pitch_link_initroll_range: Initial outward roll of hip pitch bodies.
        leg_length_range: Total leg length (thigh + shin).
        shin_ratio_range: Ratio of shin to thigh length.
        head_radius_range: Head sphere radius.
        arm_length_range: Total arm length (upper arm + forearm).
        forearm_ratio_range: forearm / upper-arm length ratio.
        valid_filter: Optional callable to filter valid parameter combinations.
    
    Example:
        >>> builder = BipedBuilder(
        ...     leg_length_range=(0.6, 1.0),
        ...     arm_length_range=(0.25, 0.45),
        ... )
        >>> param = builder.sample_params(seed=42)
        >>> spec = builder.generate_mjspec(param)
    """
    
    def __init__(
        self,
        # ── Geometry ranges ──────────────────────────────────────────────────
        torso_link_length_range:        tuple[float, float] = (0.10, 0.16),
        torso_link_width_range:         tuple[float, float] = (0.18, 0.26),
        torso_link_height_range:        tuple[float, float] = (0.20, 0.30),
        pelvis_height_range:            tuple[float, float] = (0.05, 0.08),
        pelvis_radius_coeff_range:      tuple[float, float] = (0.20, 0.40),   # × min(w, l)
        hip_spacing_range:              tuple[float, float] = (0.16, 0.24),
        hip_pitch_link_length_range:    tuple[float, float] = (0.03, 0.06),
        hip_pitch_link_radius_range:    tuple[float, float] = (0.02, 0.04),
        hip_roll_link_length_range:     tuple[float, float] = (0.03, 0.06),
        hip_roll_link_radius_range:     tuple[float, float] = (0.02, 0.04),
        hip_pitch_link_initroll_range:  tuple[float, float] = (0.00, 0.20),   # rad, ~0-11°
        hip_yaw_link_radius_range:      tuple[float, float] = (0.025, 0.040),
        leg_length_range:               tuple[float, float] = (0.50, 0.70),
        shin_ratio_range:               tuple[float, float] = (0.85, 1.15),
        ankle_roll_link_length_range:   tuple[float, float] = (0.18, 0.25),
        ankle_roll_link_width_range:    tuple[float, float] = (0.06, 0.10),
        ankle_roll_link_height_range:   tuple[float, float] = (0.02, 0.03),
        head_radius_range:              tuple[float, float] = (0.06, 0.12),
        arm_length_range:               tuple[float, float] = (0.20, 0.40),
        forearm_ratio_range:            tuple[float, float] = (0.80, 1.10),
        upper_arm_radius_range:         tuple[float, float] = (0.020, 0.040),
        # ── Mass / coefficient ranges ─────────────────────────────────────────
        torso_link_mass_range:          tuple[float, float] = (3.0,  5.0),
        hip_pitch_link_mass_range:      tuple[float, float] = (0.4,  0.8),
        hip_roll_link_mass_range:       tuple[float, float] = (0.4,  0.8),
        hip_yaw_link_mass_coeff_range:  tuple[float, float] = (1.5,  2.2),    # × length
        knee_link_radius_coeff_range:   tuple[float, float] = (0.75, 0.95),   # × hip_yaw_radius
        knee_link_mass_coeff_range:     tuple[float, float] = (1.2,  1.8),    # × length
        ankle_roll_link_mass_range:     tuple[float, float] = (0.2,  0.5),
        head_mass_coeff_range:          tuple[float, float] = (250.0, 450.0), # × radius³
        upper_arm_mass_coeff_range:     tuple[float, float] = (1.0,  2.0),    # × length
        forearm_radius_coeff_range:     tuple[float, float] = (0.75, 0.95),   # × ua_radius
        forearm_mass_coeff_range:       tuple[float, float] = (0.8,  1.5),    # × length
        # ─────────────────────────────────────────────────────────────────────
        valid_filter: Callable[[BipedParam], bool] = lambda _: True,
    ):
        super().__init__()
        # geometry
        self.torso_link_length_range       = torso_link_length_range
        self.torso_link_width_range        = torso_link_width_range
        self.torso_link_height_range       = torso_link_height_range
        self.pelvis_height_range           = pelvis_height_range
        self.pelvis_radius_coeff_range     = pelvis_radius_coeff_range
        self.hip_spacing_range             = hip_spacing_range
        self.hip_pitch_link_length_range   = hip_pitch_link_length_range
        self.hip_pitch_link_radius_range   = hip_pitch_link_radius_range
        self.hip_roll_link_length_range    = hip_roll_link_length_range
        self.hip_roll_link_radius_range    = hip_roll_link_radius_range
        self.hip_pitch_link_initroll_range = hip_pitch_link_initroll_range
        self.hip_yaw_link_radius_range     = hip_yaw_link_radius_range
        self.leg_length_range              = leg_length_range
        self.shin_ratio_range              = shin_ratio_range
        self.ankle_roll_link_length_range  = ankle_roll_link_length_range
        self.ankle_roll_link_width_range   = ankle_roll_link_width_range
        self.ankle_roll_link_height_range  = ankle_roll_link_height_range
        self.head_radius_range             = head_radius_range
        self.arm_length_range              = arm_length_range
        self.forearm_ratio_range           = forearm_ratio_range
        self.upper_arm_radius_range        = upper_arm_radius_range
        # mass / coefficients
        self.torso_link_mass_range         = torso_link_mass_range
        self.hip_pitch_link_mass_range     = hip_pitch_link_mass_range
        self.hip_roll_link_mass_range      = hip_roll_link_mass_range
        self.hip_yaw_link_mass_coeff_range = hip_yaw_link_mass_coeff_range
        self.knee_link_radius_coeff_range  = knee_link_radius_coeff_range
        self.knee_link_mass_coeff_range    = knee_link_mass_coeff_range
        self.ankle_roll_link_mass_range    = ankle_roll_link_mass_range
        self.head_mass_coeff_range         = head_mass_coeff_range
        self.upper_arm_mass_coeff_range    = upper_arm_mass_coeff_range
        self.forearm_radius_coeff_range    = forearm_radius_coeff_range
        self.forearm_mass_coeff_range      = forearm_mass_coeff_range
        self.valid_filter                  = valid_filter
    
    def sample_params(self, seed: int = -1) -> BipedParam:
        if seed >= 0:
            np.random.seed(seed)
            random.seed(seed)

        for _ in range(10):
            # Torso (box)
            torso_link_length = random.uniform(*self.torso_link_length_range)
            torso_link_width  = random.uniform(*self.torso_link_width_range)
            torso_link_height = random.uniform(*self.torso_link_height_range)
            torso_link_mass   = random.uniform(*self.torso_link_mass_range)

            # Waist connection
            pelvis_height = random.uniform(*self.pelvis_height_range)
            pelvis_radius = min(torso_link_width, torso_link_length) * random.uniform(*self.pelvis_radius_coeff_range)

            # Hip spacing and hip segment dimensions
            hip_spacing = random.uniform(*self.hip_spacing_range)

            # Hip pitch segment (first hip)
            hip_pitch_link_length = random.uniform(*self.hip_pitch_link_length_range)
            hip_pitch_link_radius = random.uniform(*self.hip_pitch_link_radius_range)
            hip_pitch_link_mass   = random.uniform(*self.hip_pitch_link_mass_range)

            # Hip roll segment (second hip)
            hip_roll_link_length = random.uniform(*self.hip_roll_link_length_range)
            hip_roll_link_radius = random.uniform(*self.hip_roll_link_radius_range)
            hip_roll_link_mass   = random.uniform(*self.hip_roll_link_mass_range)

            # Hip pitch initial roll
            hip_pitch_link_initroll = random.uniform(*self.hip_pitch_link_initroll_range)

            # Leg dimensions: split total leg into thigh (hip_yaw_link) and shin (knee_link)
            total_leg_length  = random.uniform(*self.leg_length_range)
            shin_ratio        = random.uniform(*self.shin_ratio_range)
            hip_yaw_link_length = total_leg_length / (1 + shin_ratio)
            knee_link_length    = hip_yaw_link_length * shin_ratio

            # Leg radii and masses
            hip_yaw_link_radius = random.uniform(*self.hip_yaw_link_radius_range)
            hip_yaw_link_mass   = hip_yaw_link_length * random.uniform(*self.hip_yaw_link_mass_coeff_range)
            knee_link_radius    = hip_yaw_link_radius * random.uniform(*self.knee_link_radius_coeff_range)
            knee_link_mass      = knee_link_length    * random.uniform(*self.knee_link_mass_coeff_range)

            # Foot parameters (ankle roll link)
            ankle_roll_link_length = random.uniform(*self.ankle_roll_link_length_range)
            ankle_roll_link_width  = random.uniform(*self.ankle_roll_link_width_range)
            ankle_roll_link_height = random.uniform(*self.ankle_roll_link_height_range)
            ankle_roll_link_mass   = random.uniform(*self.ankle_roll_link_mass_range)

            # Ankle pitch link (virtual body) derived from foot size
            ankle_pitch_link_radius = min(ankle_roll_link_width, ankle_roll_link_height) / 2.0
            ankle_pitch_link_length = ankle_pitch_link_radius * 2.0
            ankle_pitch_link_mass   = 0.3

            # Head
            head_radius = random.uniform(*self.head_radius_range)
            head_mass   = head_radius ** 3 * random.uniform(*self.head_mass_coeff_range)

            # Arms
            total_arm_length = random.uniform(*self.arm_length_range)
            forearm_ratio    = random.uniform(*self.forearm_ratio_range)
            upper_arm_length = total_arm_length / (1 + forearm_ratio)
            forearm_length   = upper_arm_length * forearm_ratio
            upper_arm_radius = random.uniform(*self.upper_arm_radius_range)
            upper_arm_mass   = upper_arm_length * random.uniform(*self.upper_arm_mass_coeff_range)
            forearm_radius   = upper_arm_radius * random.uniform(*self.forearm_radius_coeff_range)
            forearm_mass     = forearm_length   * random.uniform(*self.forearm_mass_coeff_range)

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
                head_radius=head_radius,
                head_mass=head_mass,
                upper_arm_length=upper_arm_length,
                upper_arm_radius=upper_arm_radius,
                upper_arm_mass=upper_arm_mass,
                forearm_length=forearm_length,
                forearm_radius=forearm_radius,
                forearm_mass=forearm_mass,
            )

            if self.valid_filter(param):
                break
        else:
            raise ValueError("Failed to sample valid parameters")

        return param
    
    def generate_mjspec(self, param: BipedParam) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()

        # ============ Torso ============
        torso_body = spec.worldbody.add_body(name="torso_link")
        torso_body.mass = param.torso_link_mass
        torso_link_ixx = param.torso_link_mass * (param.torso_link_width**2 + param.torso_link_height**2) / 12
        torso_link_iyy = param.torso_link_mass * (param.torso_link_length**2 + param.torso_link_height**2) / 12
        torso_link_izz = param.torso_link_mass * (param.torso_link_length**2 + param.torso_link_width**2) / 12
        torso_body.inertia = [torso_link_ixx, torso_link_iyy, torso_link_izz]
        torso_geom = torso_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
        torso_geom.size = [param.torso_link_length/2, param.torso_link_width/2, param.torso_link_height/2]
        torso_geom.rgba = [0.75, 0.75, 0.75, 1.0]

        # ============ Head ============
        head_link = torso_body.add_body(name="head_link")
        head_link.pos = [0, 0, param.torso_link_height/2 + param.head_radius]
        head_link.mass = param.head_mass
        head_inertia = 2.0 / 5.0 * param.head_mass * param.head_radius**2
        head_link.inertia = [head_inertia, head_inertia, head_inertia]
        head_geom = head_link.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE)
        head_geom.size = [param.head_radius, 0, 0]
        head_geom.rgba = [0.9, 0.8, 0.7, 1.0]

        # ============ Pelvis ============
        pelvis = torso_body.add_body(name="pelvis")
        pelvis.pos = [0, 0, -(param.torso_link_height/2 + param.pelvis_height/2)]
        pelvis.mass = 0.5
        pelvis.inertia = [0.01, 0.01, 0.01]
        pelvis_geom = pelvis.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER)
        pelvis_geom.size = [param.pelvis_radius, param.pelvis_height/2, 0]
        pelvis_geom.rgba = [0.65, 0.65, 0.85, 1.0]

        # ============ LEGS (6 DOF each, G1-style naming) ============
        for side, y_sign in [("left", 1), ("right", -1)]:

            # --- hip_pitch_link ---
            # IMPORTANT: Body frame is kept world-aligned (quat=identity) so that
            # hip_pitch_joint.axis=[0,1,0] always maps to world Y (sagittal plane).
            # Structural diversity from hip_pitch_link_initroll is expressed only
            # via the capsule geom direction in the parent frame, NOT via body quat.
            sin_r = np.sin(param.hip_pitch_link_initroll)
            cos_r = np.cos(param.hip_pitch_link_initroll)

            hip_pitch_link = pelvis.add_body(name=f"{side}_hip_pitch_link")
            hip_pitch_link.pos = [0, y_sign * param.hip_spacing/2, -(param.pelvis_height/2)]
            hip_pitch_link.quat = [1.0, 0.0, 0.0, 0.0]  # world-aligned: pitch joint = world Y
            hip_pitch_link.mass = param.hip_pitch_link_mass
            pitch_joint = hip_pitch_link.add_joint(
                name=f"{side}_hip_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            pitch_joint.axis = [0, 1, 0]  # Y-axis (pitch) — always world Y
            pitch_joint.range = [-2.5307, 2.8798]
            pitch_joint.armature = 0.01017752004
            hp_ixx = param.hip_pitch_link_mass * (3 * param.hip_pitch_link_radius**2 + param.hip_pitch_link_length**2) / 12
            hp_iyy = param.hip_pitch_link_mass * param.hip_pitch_link_radius**2 / 2
            hip_pitch_link.inertia = [hp_ixx, hp_iyy, hp_ixx]
            hp_geom = hip_pitch_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hp_geom.size = [param.hip_pitch_link_radius, 0, 0]
            # Geom tilted outward by initroll angle (expressed in world-aligned body frame)
            hp_geom.fromto = [
                0, 0, 0,
                0,
                y_sign * sin_r * param.hip_pitch_link_length,
                -cos_r * param.hip_pitch_link_length,
            ]
            hp_geom.rgba = [0.85, 0.45, 0.45, 1.0]

            # --- hip_roll_link ---
            # Place at the end of the (possibly tilted) hip_pitch capsule.
            # No compensation quat needed since hip_pitch_link is already world-aligned.
            hip_chain_len = param.hip_pitch_link_length + param.hip_pitch_link_radius + param.hip_roll_link_radius
            hip_roll_link = hip_pitch_link.add_body(name=f"{side}_hip_roll_link")
            hip_roll_link.pos = [
                0,
                y_sign * sin_r * hip_chain_len,
                -cos_r * hip_chain_len,
            ]
            hip_roll_link.quat = [1.0, 0.0, 0.0, 0.0]  # world-aligned: roll joint = world X
            hip_roll_link.mass = param.hip_roll_link_mass
            roll_joint = hip_roll_link.add_joint(
                name=f"{side}_hip_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            roll_joint.axis = [1, 0, 0]  # X-axis (roll)
            if side == "left":
                roll_joint.range = [-0.5236, 2.9671]
            else:
                roll_joint.range = [-2.9671, 0.5236]
            roll_joint.armature = 0.025101925
            hr_ixx = param.hip_roll_link_mass * (3 * param.hip_roll_link_radius**2 + param.hip_roll_link_length**2) / 12
            hr_iyy = param.hip_roll_link_mass * param.hip_roll_link_radius**2 / 2
            hip_roll_link.inertia = [hr_ixx, hr_iyy, hr_ixx]
            hr_geom = hip_roll_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hr_geom.size = [param.hip_roll_link_radius, 0, 0]
            hr_geom.fromto = [0, 0, 0, 0, 0, -param.hip_roll_link_length]
            hr_geom.rgba = [0.45, 0.75, 0.45, 1.0]

            # --- hip_yaw_link (thigh) ---
            hip_yaw_link = hip_roll_link.add_body(name=f"{side}_hip_yaw_link")
            hip_yaw_link.pos = [0, 0, -(param.hip_roll_link_length + param.hip_roll_link_radius + param.hip_yaw_link_radius)]
            hip_yaw_link.mass = param.hip_yaw_link_mass
            hip_yaw_link.quat = [1.0, 0.0, 0.0, 0.0]
            yaw_joint = hip_yaw_link.add_joint(
                name=f"{side}_hip_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            yaw_joint.axis = [0, 0, 1]  # Z-axis (yaw)
            yaw_joint.range = [-2.7576, 2.7576]
            yaw_joint.armature = 0.01017752004
            hy_ixx = param.hip_yaw_link_mass * (3 * param.hip_yaw_link_radius**2 + param.hip_yaw_link_length**2) / 12
            hy_iyy = param.hip_yaw_link_mass * param.hip_yaw_link_radius**2 / 2
            hip_yaw_link.inertia = [hy_ixx, hy_iyy, hy_ixx]
            hy_geom = hip_yaw_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            hy_geom.size = [param.hip_yaw_link_radius, 0, 0]
            hy_geom.fromto = [0, 0, 0, 0, 0, -param.hip_yaw_link_length]
            hy_geom.rgba = [0.45, 0.45, 0.85, 1.0]

            # --- knee_link (shin) ---
            knee_link = hip_yaw_link.add_body(name=f"{side}_knee_link")
            knee_link.pos = [0, 0, -(param.hip_yaw_link_length + param.hip_yaw_link_radius + param.knee_link_radius)]
            knee_link.mass = param.knee_link_mass
            knee_link.quat = [1.0, 0.0, 0.0, 0.0]
            knee_joint_j = knee_link.add_joint(
                name=f"{side}_knee_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            knee_joint_j.axis = [0, 1, 0]  # Y-axis (pitch)
            knee_joint_j.range = [-0.087267, 2.8798]
            knee_joint_j.armature = 0.025101925
            kl_ixx = param.knee_link_mass * (3 * param.knee_link_radius**2 + param.knee_link_length**2) / 12
            kl_iyy = param.knee_link_mass * param.knee_link_radius**2 / 2
            knee_link.inertia = [kl_ixx, kl_iyy, kl_ixx]
            kl_geom = knee_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            kl_geom.size = [param.knee_link_radius, 0, 0]
            kl_geom.fromto = [0, 0, 0, 0, 0, -param.knee_link_length]
            kl_geom.rgba = [0.85, 0.75, 0.35, 1.0]

            # --- ankle_pitch_link ---
            ankle_pitch_link = knee_link.add_body(name=f"{side}_ankle_pitch_link")
            ankle_pitch_link.pos = [0, 0, -(param.knee_link_length + param.knee_link_radius + param.ankle_pitch_link_radius)]
            ankle_pitch_link.mass = param.ankle_pitch_link_mass
            apj = ankle_pitch_link.add_joint(
                name=f"{side}_ankle_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            apj.axis = [0, 1, 0]  # Y-axis (pitch)
            apj.range = [-0.87267, 0.5236]
            apj.armature = 0.00721945
            ankle_pitch_link.inertia = [0.005, 0.005, 0.005]
            ap_geom = ankle_pitch_link.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE)
            ap_geom.size = [param.ankle_pitch_link_radius, 0, 0]
            ap_geom.rgba = [0.55, 0.85, 0.85, 1.0]

            # --- ankle_roll_link (foot) ---
            ankle_roll_link = ankle_pitch_link.add_body(name=f"{side}_ankle_roll_link")
            ankle_roll_link.pos = [0, 0, -(param.ankle_pitch_link_radius + param.ankle_roll_link_height/2)]
            ankle_roll_link.mass = param.ankle_roll_link_mass
            arj = ankle_roll_link.add_joint(
                name=f"{side}_ankle_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            arj.axis = [1, 0, 0]  # X-axis (roll)
            arj.range = [-0.2618, 0.2618]
            arj.armature = 0.00721945
            foot_ixx = param.ankle_roll_link_mass * (param.ankle_roll_link_width**2 + param.ankle_roll_link_height**2) / 12
            foot_iyy = param.ankle_roll_link_mass * (param.ankle_roll_link_length**2 + param.ankle_roll_link_height**2) / 12
            foot_izz = param.ankle_roll_link_mass * (param.ankle_roll_link_length**2 + param.ankle_roll_link_width**2) / 12
            ankle_roll_link.inertia = [foot_ixx, foot_iyy, foot_izz]
            foot_geom = ankle_roll_link.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX)
            foot_geom.size = [param.ankle_roll_link_length/2, param.ankle_roll_link_width/2, param.ankle_roll_link_height/2]
            foot_geom.pos = [param.ankle_roll_link_length/4, 0, -param.ankle_roll_link_height/2]
            foot_geom.rgba = [0.85, 0.55, 0.85, 1.0]

        # ============ ARMS (7 DOF each, G1-style naming) ============
        # Small connector segments derived from upper_arm_radius
        shoulder_conn_len = param.upper_arm_radius * 2.5
        wrist_radius = param.forearm_radius * 0.75
        wrist_seg_len = wrist_radius * 2.5

        for side, y_sign in [("left", 1), ("right", -1)]:
            # Shoulder attachment: upper portion of torso side
            s_y = y_sign * param.torso_link_width / 2
            s_z = param.torso_link_height * 0.25

            # --- shoulder_pitch_link ---
            shoulder_pitch_link = torso_body.add_body(name=f"{side}_shoulder_pitch_link")
            shoulder_pitch_link.pos = [0, s_y, s_z]
            shoulder_pitch_link.mass = param.upper_arm_mass * 0.15
            spj = shoulder_pitch_link.add_joint(
                name=f"{side}_shoulder_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            spj.axis = [0, 1, 0]  # Y-axis (pitch)
            spj.range = [-3.0892, 2.6704]
            spj.armature = 0.01017752004
            inertia_sp = shoulder_pitch_link.mass * shoulder_conn_len**2 / 12
            shoulder_pitch_link.inertia = [inertia_sp, inertia_sp, inertia_sp]
            sp_geom = shoulder_pitch_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            sp_geom.size = [param.upper_arm_radius * 0.6, 0, 0]
            sp_geom.fromto = [0, 0, 0, 0, y_sign * shoulder_conn_len, 0]
            sp_geom.rgba = [0.85, 0.55, 0.35, 1.0]

            # --- shoulder_roll_link ---
            shoulder_roll_link = shoulder_pitch_link.add_body(name=f"{side}_shoulder_roll_link")
            shoulder_roll_link.pos = [0, y_sign * shoulder_conn_len, 0]
            shoulder_roll_link.mass = param.upper_arm_mass * 0.15
            srj = shoulder_roll_link.add_joint(
                name=f"{side}_shoulder_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            srj.axis = [1, 0, 0]  # X-axis (roll)
            if side == "left":
                srj.range = [-1.5882, 2.2515]
            else:
                srj.range = [-2.2515, 1.5882]
            srj.armature = 0.025101925
            inertia_sr = shoulder_roll_link.mass * shoulder_conn_len**2 / 12
            shoulder_roll_link.inertia = [inertia_sr, inertia_sr, inertia_sr]
            sr_geom = shoulder_roll_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            sr_geom.size = [param.upper_arm_radius * 0.6, 0, 0]
            sr_geom.fromto = [0, 0, 0, 0, 0, -shoulder_conn_len]
            sr_geom.rgba = [0.35, 0.65, 0.85, 1.0]

            # --- shoulder_yaw_link (upper arm / humerus) ---
            shoulder_yaw_link = shoulder_roll_link.add_body(name=f"{side}_shoulder_yaw_link")
            shoulder_yaw_link.pos = [0, 0, -shoulder_conn_len]
            shoulder_yaw_link.mass = param.upper_arm_mass
            syj = shoulder_yaw_link.add_joint(
                name=f"{side}_shoulder_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            syj.axis = [0, 0, 1]  # Z-axis (yaw)
            syj.range = [-2.618, 2.618]
            syj.armature = 0.01017752004
            ua_ixx = param.upper_arm_mass * (3 * param.upper_arm_radius**2 + param.upper_arm_length**2) / 12
            ua_iyy = param.upper_arm_mass * param.upper_arm_radius**2 / 2
            shoulder_yaw_link.inertia = [ua_ixx, ua_iyy, ua_ixx]
            ua_geom = shoulder_yaw_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            ua_geom.size = [param.upper_arm_radius, 0, 0]
            ua_geom.fromto = [0, 0, 0, 0, 0, -param.upper_arm_length]
            ua_geom.rgba = [0.55, 0.85, 0.45, 1.0]

            # --- elbow_link (forearm) ---
            elbow_link = shoulder_yaw_link.add_body(name=f"{side}_elbow_link")
            elbow_link.pos = [0, 0, -(param.upper_arm_length + param.upper_arm_radius + param.forearm_radius)]
            elbow_link.mass = param.forearm_mass
            ej = elbow_link.add_joint(
                name=f"{side}_elbow_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            ej.axis = [0, 1, 0]  # Y-axis (pitch)
            ej.range = [-1.0472, 2.0944]
            ej.armature = 0.025101925
            fa_ixx = param.forearm_mass * (3 * param.forearm_radius**2 + param.forearm_length**2) / 12
            fa_iyy = param.forearm_mass * param.forearm_radius**2 / 2
            elbow_link.inertia = [fa_ixx, fa_iyy, fa_ixx]
            fa_geom = elbow_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            fa_geom.size = [param.forearm_radius, 0, 0]
            fa_geom.fromto = [0, 0, 0, 0, 0, -param.forearm_length]
            fa_geom.rgba = [0.45, 0.85, 0.75, 1.0]

            # --- wrist_roll_link ---
            wrist_roll_link = elbow_link.add_body(name=f"{side}_wrist_roll_link")
            wrist_roll_link.pos = [0, 0, -(param.forearm_length + param.forearm_radius + wrist_radius)]
            wrist_roll_link.mass = param.forearm_mass * 0.1
            wrj = wrist_roll_link.add_joint(
                name=f"{side}_wrist_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            wrj.axis = [1, 0, 0]  # X-axis (roll)
            wrj.range = [-1.972222054, 1.972222054]
            wrj.armature = 0.00721945
            inertia_wr = wrist_roll_link.mass * wrist_seg_len**2 / 12
            wrist_roll_link.inertia = [inertia_wr, inertia_wr, inertia_wr]
            wr_geom = wrist_roll_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            wr_geom.size = [wrist_radius, 0, 0]
            wr_geom.fromto = [0, 0, 0, 0, 0, -wrist_seg_len]
            wr_geom.rgba = [0.85, 0.85, 0.45, 1.0]

            # --- wrist_pitch_link ---
            wrist_pitch_link = wrist_roll_link.add_body(name=f"{side}_wrist_pitch_link")
            wrist_pitch_link.pos = [0, 0, -wrist_seg_len]
            wrist_pitch_link.mass = param.forearm_mass * 0.1
            wpj = wrist_pitch_link.add_joint(
                name=f"{side}_wrist_pitch_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            wpj.axis = [0, 1, 0]  # Y-axis (pitch)
            wpj.range = [-1.614429558, 1.614429558]
            wpj.armature = 0.00721945
            inertia_wp = wrist_pitch_link.mass * wrist_seg_len**2 / 12
            wrist_pitch_link.inertia = [inertia_wp, inertia_wp, inertia_wp]
            wp_geom = wrist_pitch_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            wp_geom.size = [wrist_radius, 0, 0]
            wp_geom.fromto = [0, 0, 0, 0, 0, -wrist_seg_len]
            wp_geom.rgba = [0.85, 0.65, 0.45, 1.0]

            # --- wrist_yaw_link ---
            wrist_yaw_link = wrist_pitch_link.add_body(name=f"{side}_wrist_yaw_link")
            wrist_yaw_link.pos = [0, 0, -wrist_seg_len]
            wrist_yaw_link.mass = param.forearm_mass * 0.15
            wyj = wrist_yaw_link.add_joint(
                name=f"{side}_wrist_yaw_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
            )
            wyj.axis = [0, 0, 1]  # Z-axis (yaw)
            wyj.range = [-1.614429558, 1.614429558]
            wyj.armature = 0.00721945
            inertia_wy = wrist_yaw_link.mass * wrist_seg_len**2 / 12
            wrist_yaw_link.inertia = [inertia_wy, inertia_wy, inertia_wy]
            wy_geom = wrist_yaw_link.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
            wy_geom.size = [wrist_radius, 0, 0]
            wy_geom.fromto = [0, 0, 0, 0, 0, -wrist_seg_len]
            wy_geom.rgba = [0.75, 0.55, 0.85, 1.0]

        return spec

    # =========================================================================
    # Standing-pose helpers
    # =========================================================================

    @staticmethod
    def _compute_standing_pose(param: BipedParam) -> tuple[dict, float]:
        """Compute default joint angles and standing height for a biped morphology.

        Kinematics (sagittal plane, angles from the vertical -Z axis, positive = forward):
          - θ_thigh = ``hip_pitch_joint``
          - θ_shin  = ``hip_pitch_joint`` + ``knee_joint``
              (positive knee rotates shin forward relative to thigh per G1 convention)
          - ``ankle_pitch_joint`` = -(θ_thigh + knee) keeps foot horizontal

        Foot-under-hip constraint  (x_foot ≈ 0):
          L1·sin(θ_thigh) + L2·sin(θ_thigh + knee) = 0
          ⟹  sin(θ_thigh + knee) = -(L1/L2)·sin(θ_thigh)

        Returns:
            ({'hip_pitch': rad, 'knee': rad, 'ankle_pitch': rad},
             standing_height_meters)
        """
        l_thigh = param.hip_yaw_link_length
        l_shin  = param.knee_link_length

        # Slight forward lean for a stable crouch
        hip_pitch = 0.20  # rad (~11.5°)

        # Knee angle: place foot under hip
        sin_sum = -(l_thigh / l_shin) * np.sin(hip_pitch)
        sin_sum = np.clip(sin_sum, -1.0, 1.0)
        knee = float(np.clip(np.arcsin(sin_sum) - hip_pitch, 0.05, 1.20))

        # Ankle: compensate to keep foot flat
        ankle_pitch = float(np.clip(-(hip_pitch + knee), -0.87267, 0.5236))

        # ---- Compute standing height ----------------------------------------
        cos_r = np.cos(param.hip_pitch_link_initroll)
        hp_r  = param.hip_pitch_link_radius
        hr_r  = param.hip_roll_link_radius
        hy_r  = param.hip_yaw_link_radius
        kl_r  = param.knee_link_radius
        ap_r  = param.ankle_pitch_link_radius
        ah    = param.ankle_roll_link_height

        # Vertical distance from torso centre → thigh pivot
        # chain: torso_h/2 → pelvis → hip_pitch_link (tilted) → hip_roll_link → hip_yaw origin
        z_to_thigh_pivot = (
            param.torso_link_height / 2.0
            + param.pelvis_height
            + cos_r * (param.hip_pitch_link_length + hp_r + hr_r)   # hip_pitch capsule + gap
            + (param.hip_roll_link_length + hr_r + hy_r)             # hip_roll capsule + gap
        )

        # Vertical contribution of the thigh and shin (bent by joint angles)
        z_thigh = l_thigh * np.cos(hip_pitch) + (hy_r + kl_r)        # thigh + gap at knee
        z_shin  = l_shin  * np.cos(hip_pitch + knee) + (kl_r + ap_r)  # shin + gap at ankle

        # Ankle capsule → foot centre → foot bottom
        z_ankle = ap_r + ah / 2.0 + ah / 2.0

        standing_height = float(z_to_thigh_pivot + z_thigh + z_shin + z_ankle)

        return {
            "hip_pitch":   hip_pitch,
            "knee":        knee,
            "ankle_pitch": ankle_pitch,
        }, standing_height

    # =========================================================================
    # Articulation modifier (call from env __init__ after super().__init__())
    # =========================================================================

    @classmethod
    def modify_articulation(cls, articulation: "Articulation") -> None:
        """Set per-robot default joint positions and root heights.

        Mirrors ``QuadrupedBuilder.modify_articulation``.  Should be called
        once from the env ``__init__`` after ``super().__init__()``.

        Sets:
          - ``hip_pitch_joint`` / ``knee_joint`` / ``ankle_pitch_joint``
            to the computed standing pose for each robot.
          - Root Z to the per-robot standing height.
        """
        builder: BipedBuilder = cls.get_instance()
        device = articulation.device

        hip_pitch_joint_ids   = articulation.find_joints(".*_hip_pitch_joint")[0]
        knee_joint_ids        = articulation.find_joints(".*_knee_joint")[0]
        ankle_pitch_joint_ids = articulation.find_joints(".*_ankle_pitch_joint")[0]

        default_joint_pos = articulation.data.default_joint_pos.clone()
        heights: list[float] = []

        for i, param in enumerate(builder.params):
            angles, height = cls._compute_standing_pose(param)
            heights.append(height)
            default_joint_pos[i, hip_pitch_joint_ids]   = angles["hip_pitch"]
            default_joint_pos[i, knee_joint_ids]         = angles["knee"]
            default_joint_pos[i, ankle_pitch_joint_ids]  = angles["ankle_pitch"]

        articulation.data.default_joint_pos[:] = default_joint_pos
        articulation.write_joint_state_to_sim(
            default_joint_pos, torch.zeros_like(default_joint_pos)
        )

        # Per-robot initial root height
        default_root_state = articulation.data.default_root_state.clone()
        for i, h in enumerate(heights):
            default_root_state[i, 2] = h
        articulation.data.default_root_state[:] = default_root_state
        articulation.write_root_state_to_sim(default_root_state)

        print(
            f"[INFO] Procedural biped standing poses computed: "
            f"h_min={min(heights):.3f}m  h_max={max(heights):.3f}m  "
            f"h_mean={sum(heights)/len(heights):.3f}m"
        )
        print(
            f"       env0: hip_pitch={default_joint_pos[0, hip_pitch_joint_ids[0]]:.3f}  "
            f"knee={default_joint_pos[0, knee_joint_ids[0]]:.3f}  "
            f"ankle_pitch={default_joint_pos[0, ankle_pitch_joint_ids[0]]:.3f}  "
            f"h={heights[0]:.3f}m"
        )


if __name__ == "__main__":
    builder = BipedBuilder()
    param = builder.sample_params(seed=0)
    print(param)
    spec = builder.generate_mjspec(param)
    print("Generated biped MjSpec successfully!")
