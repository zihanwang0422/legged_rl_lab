import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Example script for procedural biped generation.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, ContactSensor
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from metamorphosis.asset_cfg import ProceduralBipedCfg, BipedBuilder


BIPED_CONFIG = ArticulationCfg(
    spawn=ProceduralBipedCfg(
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        # Torso parameters
            torso_link_length_range=(0.08, 0.14),
            torso_link_width_range=(0.14, 0.20),
            torso_link_height_range=(0.30, 0.6),  
        pelvis_height_range=(0.04, 0.06),
        # Hip spacing and hip segment dimensions
        hip_spacing_range=(0.12, 0.16),
        hip_pitch_link_length_range=(0.025, 0.045),
        hip_pitch_link_radius_range=(0.015, 0.035),
        hip_roll_link_length_range=(0.025, 0.045),
        hip_roll_link_radius_range=(0.015, 0.035),
        hip_pitch_link_initroll_range=(0.0, np.pi / 2),
        # Leg lengths - closer to human proportions
        leg_length_range=(0.4, 0.6),
        # Leg proportions
        shin_ratio_range=(0.85, 1.15),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Hip joints - G1-style neutral position
            ".*_hip_pitch_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            # Knee - straight for standing
            ".*_knee_joint": 0.0,
            # Ankle
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
        },
        pos=(0, 0, 1.2),  # Higher initial position to avoid ground collision
    ),
    actuators={
        # Hip actuators - G1-style configuration
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            effort_limit_sim=88.0,  # G1 spec: actuatorfrcrange="-88 88"
            stiffness=75.0,         # G1 default kp=75
            damping=2.0,           # G1 default kv=2
            armature=0.01017752004,
            friction=0.1,          # G1 default frictionloss=0.1
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            effort_limit_sim=139.0, # G1 spec: actuatorfrcrange="-139 139"
            stiffness=75.0,
            damping=2.0,
            armature=0.025101925,
            friction=0.1,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            effort_limit_sim=88.0,  # G1 spec: actuatorfrcrange="-88 88"
            stiffness=75.0,
            damping=2.0,
            armature=0.01017752004,
            friction=0.1,
        ),
        # Knee actuators
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit_sim=139.0, # G1 spec: actuatorfrcrange="-139 139"
            stiffness=75.0,
            damping=2.0,
            armature=0.025101925,
            friction=0.1,
        ),
        # Ankle actuators - lower gains for stability
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=50.0,  # G1 spec: actuatorfrcrange="-50 50"
            stiffness=20.0,        # G1 ankle kp=20
            damping=2.0,          # G1 ankle kv=2
            armature=0.00721945,
            friction=0.1,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.00721945,
            friction=0.1,
        ),
    },
)


class ProceduralBipedSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Biped robot
    biped = BIPED_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Biped")
    
    # Contact sensor for feet
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Biped/.*",
        track_air_time=True,
        history_length=3,
    )


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.0, 0.0, 2.0], [0.0, 0.0, 1.0])
    
    scene_cfg = ProceduralBipedSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
        replicate_physics=False  # Important for morphology randomization!
    )
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    scene.reset()
    
    articulation = scene.articulations["biped"]
    contact_sensor: ContactSensor = scene.sensors["contact_sensor"]

    print("=" * 60)
    print("Procedural Biped Robot Info")
    print("=" * 60)
    print(f"Number of instances: {articulation.num_instances}")
    print(f"Joint names: {articulation.joint_names}")
    print(f"Body names: {articulation.body_names}")
    print(f"Is homogeneous: {articulation.root_physx_view.is_homogeneous}")
    
    builder = BipedBuilder.get_instance()
    print("\nGenerated parameters for each environment:")
    for i, param in enumerate(builder.params[:min(5, len(builder.params))]):  # Show first 5 only
        leg_len = param.hip_yaw_link_length + param.knee_link_length
        print(
                f"  Env {i}: torso=({param.torso_link_length:.3f}×{param.torso_link_width:.3f}×{param.torso_link_height:.3f}), "
            f"hip_spacing={param.hip_spacing:.3f}, leg_length={leg_len:.2f}, "
            f"hip_pitch_link=(r={param.hip_pitch_link_radius:.3f}, l={param.hip_pitch_link_length:.3f}), "
            f"hip_roll_link=(r={param.hip_roll_link_radius:.3f}, l={param.hip_roll_link_length:.3f})"
        )
    if len(builder.params) > 5:
        print(f"  ... and {len(builder.params) - 5} more environments")
    print("=" * 60)

    root_state = articulation.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    default_joint_pos = articulation.data.default_joint_pos.clone()


    articulation.write_root_pose_to_sim(root_state[:, :7])
    articulation.write_joint_position_to_sim(default_joint_pos)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    step_count = 0
    while simulation_app.is_running():
        # perform step
        articulation.set_joint_position_target(default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        
        # Print contact info periodically
        step_count += 1
        if step_count % 100 == 0:
            foot_forces = contact_sensor.data.net_forces_w
            print(f"Step {step_count}: Contact forces shape = {foot_forces.shape}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
