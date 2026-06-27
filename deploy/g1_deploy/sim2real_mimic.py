#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np

from common.command_helper import create_damping_cmd
from common.remote_controller import KeyMap
from config import Config
from sim2real_walk import Controller
from sim2sim_mimic import (
    build_tracking_obs,
    matrix_from_quat_np,
    subtract_frame_transforms_np,
)
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.utils.crc import CRC


THIS_DIR = Path(__file__).resolve().parent


def resolve_config(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    return THIS_DIR / "config" / path


def resolve_model(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    return THIS_DIR / "exported_policy" / name


class MimicController(Controller):
    def __init__(self, config: Config, net: str, domain_id: int = 0, debug_policy: bool = False) -> None:
        self.time_step = 0
        self.ref_joint_pos = None
        self.ref_joint_vel = None
        self.ref_body_pos_w = None
        self.ref_body_quat_w = None
        self.sport_state = unitree_go_msg_dds__SportModeState_()
        self.sport_state_received = False
        super().__init__(config, net, domain_id=domain_id, debug_policy=debug_policy)

    def setup_extra_subscribers(self):
        self.sport_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sport_subscriber.Init(self.SportStateHandler, 10)

    def SportStateHandler(self, msg: SportModeState_):
        self.sport_state = msg
        self.sport_state_received = True

    def infer_policy(self, obs_batch):
        obs = obs_batch.reshape(1, -1).astype(np.float32)
        if obs.shape[1] != self.config.num_obs:
            obs = np.zeros((1, self.config.num_obs), dtype=np.float32)
        ts = np.array([[self.time_step]], dtype=np.float32)
        outputs = self.policy.run(self.policy_output_names, {"obs": obs, "time_step": ts})
        action = outputs[0].reshape(-1).astype(np.float32)
        if len(outputs) >= 5:
            self.ref_joint_pos = outputs[1].squeeze(0).astype(np.float32)
            self.ref_joint_vel = outputs[2].squeeze(0).astype(np.float32)
            self.ref_body_pos_w = outputs[3].squeeze(0).astype(np.float32)
            self.ref_body_quat_w = outputs[4].squeeze(0).astype(np.float32)
        return action

    def run(self):
        for i in range(len(self.config.sdk2isaac_idx)):
            self.joint_pos[i] = self.low_state.motor_state[self.config.sdk2isaac_idx[i]].q
            self.joint_vel[i] = self.low_state.motor_state[self.config.sdk2isaac_idx[i]].dq

        if self.ref_joint_pos is None:
            self.infer_policy(np.zeros((1, self.config.num_obs), dtype=np.float32))

        joint_pos_rel = (self.joint_pos - self.config.default_joint_pos).astype(np.float32)
        joint_vel = self.joint_vel.astype(np.float32)
        last_action = self.action.astype(np.float32)

        base_quat_wxyz = np.asarray(self.low_state.imu_state.quaternion, dtype=np.float32)
        base_ang_vel_b = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        base_lin_vel_b = np.asarray(self.sport_state.velocity, dtype=np.float32) if self.sport_state_received else np.zeros(3, dtype=np.float32)
        robot_anchor_pos = np.asarray(self.sport_state.position, dtype=np.float32) if self.sport_state_received else np.zeros(3, dtype=np.float32)
        robot_anchor_quat = base_quat_wxyz

        anchor_idx = self.config.body_names.index(self.config.anchor_body_name)
        motion_anchor_pos = self.ref_body_pos_w[anchor_idx].astype(np.float32)
        motion_anchor_quat = self.ref_body_quat_w[anchor_idx].astype(np.float32)
        anchor_pos_b, anchor_quat_b = subtract_frame_transforms_np(
            robot_anchor_pos, robot_anchor_quat, motion_anchor_pos, motion_anchor_quat
        )
        anchor_mat = matrix_from_quat_np(anchor_quat_b)
        anchor_ori_b_6 = anchor_mat[:, :2].reshape(-1).astype(np.float32)

        obs = build_tracking_obs(
            self.ref_joint_pos.astype(np.float32),
            self.ref_joint_vel.astype(np.float32),
            anchor_pos_b.astype(np.float32),
            anchor_ori_b_6,
            base_lin_vel_b,
            base_ang_vel_b,
            joint_pos_rel,
            joint_vel,
            last_action,
            include_state_estimation=bool(self.config.include_state_estimation),
        )

        raw_action = self.infer_policy(obs.reshape(1, -1))
        self.action = np.clip(raw_action, -self.config.action_clip, self.config.action_clip)
        target_dof_pos = self.config.default_joint_pos + self.action * self.config.action_scale

        ramp = 1.0
        if self.config.policy_ramp_time > 0.0:
            ramp = np.clip((time.time() - self.control_start_time) / self.config.policy_ramp_time, 0.0, 1.0)
            target_dof_pos = self.config.default_joint_pos + ramp * (target_dof_pos - self.config.default_joint_pos)

        self.print_policy_debug(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), base_ang_vel_b, joint_pos_rel, joint_vel, target_dof_pos, raw_action, ramp)
        with self.cmd_lock:
            for i, motor_idx in enumerate(self.config.sdk2isaac_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])

        self.time_step += 1
        total_steps = getattr(self.config, "motion_total_steps", None)
        if total_steps and self.time_step >= total_steps:
            self.time_step = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="SDK2 sim2real controller for G1 mimic/tracking policies.")
    parser.add_argument("--net", type=str, default="enp108s0", help="network interface")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain id, use 1 for local SDK2 MuJoCo bridge")
    parser.add_argument("--config_path", type=str, default="config/g1_mimic.yaml", help="configuration file path")
    parser.add_argument("--model", type=str, default="g1_dance.onnx", help="ONNX model filename or path")
    parser.add_argument("--debug_policy", action="store_true", help="Print policy observation/action ranges.")
    args = parser.parse_args()

    config = Config(resolve_config(args.config_path))
    config.set_policy_path(str(resolve_model(args.model)))
    controller = MimicController(config, args.net, args.domain_id, debug_policy=args.debug_policy)

    try:
        while True:
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("Select Button detected, Exit!")
                break
            time.sleep(0.01)
    finally:
        controller.run_thread.Wait()
        controller.publish_thread.Wait()
        with controller.cmd_lock:
            controller.low_cmd = create_damping_cmd(controller.low_cmd)
            controller.low_cmd.crc = CRC().Crc(controller.low_cmd)
            controller.lowcmd_publisher_.Write(controller.low_cmd)
        time.sleep(0.2)
        print("Exit")


if __name__ == "__main__":
    main()
