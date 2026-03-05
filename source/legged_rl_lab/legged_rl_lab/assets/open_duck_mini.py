## Copyright (c) 2025, Giulio Romualdi
## BSD 3-Clause License


"""Configuration helpers for the open_duck_mini robot."""


from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

OPEN_DUCK_MINI_ACTUATED_JOINTS = [
    "left_hip_yaw",
    "neck_pitch",
    "right_hip_yaw",
    "left_hip_roll",
    "head_pitch",
    "right_hip_roll",
    "left_hip_pitch",
    "head_yaw",
    "right_hip_pitch",
    "left_knee",
    "head_roll",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

OPEN_DUCK_MINI_NOT_ACTUATED_JOINTS = ["left_antenna", "right_antenna"]

# Derived directly from playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash.xml.
OPEN_DUCK_MINI_JOINT_LIMITS = {
    "left_hip_yaw": (-0.523598775598298, 0.5235987755983),
    "left_hip_roll": (-0.436332312998581, 0.436332312998583),
    "left_hip_pitch": (-1.221730476396031, 0.523598775598299),
    "left_knee": (-1.570796326794897, 1.570796326794897),
    "left_ankle": (-1.570796326794896, 1.570796326794897),
    "neck_pitch": (-0.349065850398844, 1.134464013796336),
    "head_pitch": (-0.785398163397448, 0.785398163397448),
    "head_yaw": (-2.792526803190927, 2.792526803190927),
    "head_roll": (-0.523598775598218, 0.52359877559838),
    "right_hip_yaw": (-0.523598775598297, 0.523598775598301),
    "right_hip_roll": (-0.43633231299858, 0.436332312998585),
    "right_hip_pitch": (-0.523598775598299, 1.221730476396031),
    "right_knee": (-1.570796326794897, 1.570796326794897),
    "right_ankle": (-1.570796326794896, 1.570796326794897),
}


@dataclass(frozen=True)
class OpenDuckMiniActuatorSpecs:
    """Aggregated actuator constants extracted from the MuJoCo XML."""

    stiffness: float = 17.8
    damping: float = 0.0
    effort_limit: float = 3.35
    friction: float = 0.052
    viscous_friction: float = 0.6
    armature: float = 0.028


OPEN_DUCK_MINI_ACTUATOR_SPECS = OpenDuckMiniActuatorSpecs()


class OpenDuckMiniCfgBuilder:
    @staticmethod
    def build_robot_cfg(model_path_relative: Path):
        cfg_path = Path(__file__).parent / "models"
        model_path = cfg_path / model_path_relative
        robot_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(model_path.resolve()),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.22),
            ),
            soft_joint_pos_limit_factor=0.95,
            actuators={
                "open_duck_mini": IdealPDActuatorCfg(
                    joint_names_expr=OPEN_DUCK_MINI_ACTUATED_JOINTS,
                    stiffness=OPEN_DUCK_MINI_ACTUATOR_SPECS.stiffness,
                    damping=OPEN_DUCK_MINI_ACTUATOR_SPECS.damping,
                    effort_limit=OPEN_DUCK_MINI_ACTUATOR_SPECS.effort_limit,
                    friction=OPEN_DUCK_MINI_ACTUATOR_SPECS.friction,
                    dynamic_friction=OPEN_DUCK_MINI_ACTUATOR_SPECS.friction,
                    viscous_friction=OPEN_DUCK_MINI_ACTUATOR_SPECS.viscous_friction,
                    armature=OPEN_DUCK_MINI_ACTUATOR_SPECS.armature,
                ),
                "antenna": ImplicitActuatorCfg(
                    joint_names_expr=OPEN_DUCK_MINI_NOT_ACTUATED_JOINTS,
                    stiffness=5.0,
                    damping=0.3,
                ),
            },
        )
        return robot_cfg, model_path


OPEN_DUCK_MINI_CFG, OPEN_DUCK_MINI_MODEL_PATH = OpenDuckMiniCfgBuilder.build_robot_cfg(
    Path("open_duck_mini") / "model.usd"
)