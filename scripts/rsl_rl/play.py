# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

RSL_RL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../source/legged_rl_lab/legged_rl_lab/rsl_rl")
)
if RSL_RL_PATH not in sys.path:
    sys.path.insert(0, RSL_RL_PATH)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--vis_attention", action="store_true", default=False, help="Visualize attention weights as colored markers (env 0).")
parser.add_argument("--motion_file", type=str, default=None, help="Path to motion NPZ file (required for Tracking tasks).")
parser.add_argument("--ckpt", type=str, default=None, help="Name of a checkpoint file under the ckpt/ directory (e.g. 'model.pt').")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

from rsl_rl.runners import DistillationRunner, OnPolicyRunner, TsDepthRunner
from legged_rl_lab.tasks.tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import legged_rl_lab  # noqa: F401 - Register custom environments
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # inject motion file for Tracking tasks
    if args_cli.motion_file is not None and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.ckpt:
        ckpt_path = os.path.join("ckpt", args_cli.ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        resume_path = os.path.abspath(ckpt_path)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent_dict = agent_cfg.to_dict()
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, cli_args.convert_policy_cfg_to_actor_critic(agent_dict), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "TsDepthRunner":
        runner = TsDepthRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if agent_cfg.class_name == "TsDepthRunner":
        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        timestep = 0
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
        env.close()
        return

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        try:
            # version 2.2 and below
            policy_nn = runner.alg.actor_critic
        except AttributeError:
            # current split actor/critic stack
            policy_nn = runner.alg.actor

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    elif hasattr(policy_nn, "obs_normalizer"):
        normalizer = policy_nn.obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if args_cli.motion_file is not None:
        export_motion_policy_as_onnx(
            env.unwrapped, policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
        )
        attach_onnx_metadata(env.unwrapped, log_dir, export_model_dir)
    elif hasattr(policy_nn, "as_jit") and hasattr(policy_nn, "as_onnx"):
        runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    robot_asset = env.unwrapped.scene["robot"]

    # --- attention viz setup ---
    attention_visualizer = None
    identity_quat = None
    attention_num_bins = 8

    def _infer_scan_grid_shape():
        if hasattr(policy_nn, "width") and hasattr(policy_nn, "length"):
            rows = int(getattr(policy_nn, "width"))
            cols = int(getattr(policy_nn, "length"))
            if rows > 0 and cols > 0:
                return rows, cols
        try:
            pattern_cfg = env_cfg.scene.height_scanner.pattern_cfg
            resolution = float(pattern_cfg.resolution)
            size_x, size_y = pattern_cfg.size
            cols = int(round(float(size_x) / resolution)) + 1
            rows = int(round(float(size_y) / resolution)) + 1
            if rows > 0 and cols > 0:
                return rows, cols
        except Exception:
            return None
        return None

    scan_grid_shape = _infer_scan_grid_shape()

    def _build_attention_visualizer(device):
        nonlocal identity_quat
        colors = []
        for i in range(attention_num_bins):
            t = i / max(1, attention_num_bins - 1)
            colors.append((t, 0.0, 1.0 - t))
        markers = {
            f"dot_{i}": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[i]),
            )
            for i in range(attention_num_bins)
        }
        cfg = VisualizationMarkersCfg(prim_path="/Visuals/attention", markers=markers)
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        return VisualizationMarkers(cfg)

    def _vis_attention_on_terrain(attn, env_obj, step):
        if attn is None or attn.numel() == 0:
            return
        try:
            height_scanner = env_obj.unwrapped.scene["height_scanner"]
            if not hasattr(height_scanner.data, "ray_hits_w"):
                return
            ray_hits = height_scanner.data.ray_hits_w[0]
            attn0 = attn[0]
            if attn0.dim() > 1:
                attn0 = attn0.reshape(-1)
            n_attn = int(attn0.shape[0])
            n_rays = int(ray_hits.shape[0])

            if n_rays == n_attn:
                selected_hits = ray_hits
                attn_vals = attn0
            elif n_rays > n_attn:
                selected_hits = None
                if scan_grid_shape is not None:
                    rows, cols = scan_grid_shape
                    if rows * cols == n_rays:
                        ray_hits_grid = ray_hits.reshape(rows, cols, 3)
                        if hasattr(policy_nn, "cnn_downsample") and bool(getattr(policy_nn, "cnn_downsample")):
                            ds_hits = ray_hits_grid[::2, ::2, :].reshape(-1, 3)
                            if ds_hits.shape[0] == n_attn:
                                selected_hits = ds_hits
                if selected_hits is None:
                    sample_idx = torch.linspace(0, n_rays - 1, steps=n_attn, device=ray_hits.device).round().long()
                    selected_hits = ray_hits[sample_idx]
                attn_vals = attn0
            else:
                selected_hits = ray_hits
                attn_vals = attn0[:n_rays]

            valid_mask = (selected_hits[:, 2] > -50.0) & (selected_hits[:, 2] < 100.0)
            if not valid_mask.any():
                return
            selected_hits = selected_hits[valid_mask]
            attn_vals = attn_vals[valid_mask]

            attn_norm = (attn_vals - attn_vals.min()) / (attn_vals.max() - attn_vals.min() + 1e-6)
            n_vis = selected_hits.shape[0]
            bins = (attn_norm * (attention_num_bins - 1)).long().clamp(0, attention_num_bins - 1)

            points = selected_hits.clone()
            points[:, 2] += 0.05
            scale_factors = attn_norm * (3.0 - 0.5) + 0.5
            scales = scale_factors.unsqueeze(1).expand(-1, 3)

            orientations = identity_quat.unsqueeze(0).expand(n_vis, -1)
            attention_visualizer.visualize(points, orientations, marker_indices=bins, scales=scales)
        except Exception as e:
            if step % 100 == 0:
                print(f"[WARN] Attention terrain viz failed: {e}")

    if args_cli.vis_attention:
        attention_visualizer = _build_attention_visualizer(env.unwrapped.device)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            # visualize attention weights from the last forward pass
            if args_cli.vis_attention and attention_visualizer is not None:
                attn_w = getattr(policy_nn, "last_attention_weights", None)
                if attn_w is not None:
                    _vis_attention_on_terrain(attn_w, env, timestep)
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
