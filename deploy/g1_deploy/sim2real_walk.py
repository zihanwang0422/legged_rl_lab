# Copyright (c) 2022-2025, The unitree_rl_gym Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from unitree_rl_gym Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import sys
import time
import os
from pathlib import Path
from threading import Lock

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = THIS_DIR.parent


def _bootstrap_cyclonedds_home() -> None:
    """Ensure CYCLONEDDS_HOME points to a valid local install before SDK import."""
    env_home = os.environ.get("CYCLONEDDS_HOME", "")
    if env_home and (Path(env_home) / "lib" / "libddsc.so").exists():
        return

    candidates = [
        THIS_DIR / "cyclonedds" / "install",
        DEPLOY_DIR / "cyclonedds" / "install",
    ]
    for candidate in candidates:
        if (candidate / "lib" / "libddsc.so").exists():
            os.environ["CYCLONEDDS_HOME"] = str(candidate)
            print(f"[CycloneDDS] Using CYCLONEDDS_HOME={candidate}")
            return


_bootstrap_cyclonedds_home()

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.command_helper import (
    MotorMode,
    create_damping_cmd,
    init_cmd_go,
    init_cmd_hg,
)
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from config import Config


class Controller:
    def __init__(self, config: Config, net: str) -> None:

        ChannelFactoryInitialize(0, net)

        self.first_run = True
        self.config = config
        self.remote_controller = RemoteController()

        self.policy = torch.jit.load(config.policy_path).eval()
        self.run_thread = RecurrentThread(interval=self.config.control_dt, target=self.run)  # 100Hz/50Hz
        self.publish_thread = RecurrentThread(interval=1 / 500, target=self.publish)
        self.cmd_lock = Lock()

        self.joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        self.current_obs = np.zeros(config.num_obs, dtype=np.float32)
        self.current_obs_history = np.zeros((config.history_length, config.num_obs), dtype=np.float32)

        self.clip_min_command = np.array(
            [
                self.config.command_range["lin_vel_x"][0],
                self.config.command_range["lin_vel_y"][0],
                self.config.command_range["ang_vel_z"][0],
            ],
            dtype=np.float32,
        )
        self.clip_max_command = np.array(
            [
                self.config.command_range["lin_vel_x"][1],
                self.config.command_range["lin_vel_y"][1],
                self.config.command_range["ang_vel_z"][1],
            ],
            dtype=np.float32,
        )

        for _ in range(50):
            with torch.inference_mode():
                obs = self.current_obs_history.reshape(1, -1).astype(np.float32)
                self.policy(torch.from_numpy(obs))

        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        else:
            raise ValueError("Invalid msg_type")

        self.wait_for_low_state()

        if config.msg_type == "hg":
            self.low_cmd = init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            self.low_cmd = init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.publish_thread.Start()
        self.wait_for_start()

        self.move_to_default_pos()
        self.wait_for_control()

        print("Start Control!")
        self.run_thread.Start()

    def LowStateHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def publish(self):
        with self.cmd_lock:
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

    def stop(self):
        print("Select Button detected, Exit!")
        self.publish_thread.Wait()
        with self.cmd_lock:
            self.low_cmd = create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
        time.sleep(0.2)
        sys.exit(0)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        self.mode_machine_ = self.low_state.mode_machine
        print("Successfully connected to the robot.")

    def wait_for_start(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal to move to default pos...")
        while self.remote_controller.button[KeyMap.start] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.sdk2isaac_idx
        dof_size = len(dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        for i in range(num_step):
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            alpha = i / num_step
            with self.cmd_lock:
                for j in range(dof_size):
                    motor_idx = dof_idx[j]
                    target_pos = self.config.default_joint_pos[j]
                    self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                    self.low_cmd.motor_cmd[motor_idx].dq = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[j]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[j]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
            time.sleep(self.config.control_dt)

    def wait_for_control(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal to Start Control...")
        while self.remote_controller.button[KeyMap.A] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def run(self):
        for i in range(len(self.config.sdk2isaac_idx)):
            self.joint_pos[i] = self.low_state.motor_state[self.config.sdk2isaac_idx[i]].q
            self.joint_vel[i] = self.low_state.motor_state[self.config.sdk2isaac_idx[i]].dq

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = self.low_state.motor_state[self.config.torso_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel
            )

        gravity_orientation = get_gravity_orientation(quat)
        joint_pos = (self.joint_pos - self.config.default_joint_pos) * self.config.dof_pos_scale
        joint_vel = self.joint_vel * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        command = np.array(
            [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx], dtype=np.float32
        )
        command *= self.config.command_scale
        command = np.clip(command, self.clip_min_command, self.clip_max_command)

        num_actions = self.config.num_actions
        self.current_obs[:3] = ang_vel
        self.current_obs[3:6] = gravity_orientation
        self.current_obs[6:9] = command
        self.current_obs[9 : 9 + num_actions] = joint_pos
        self.current_obs[9 + num_actions : 9 + num_actions * 2] = joint_vel
        self.current_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action

        if self.first_run:
            self.current_obs_history[:] = self.current_obs.reshape(1, -1)
            self.first_run = False
        else:
            self.current_obs_history = np.concatenate(
                (self.current_obs_history[1:], self.current_obs.reshape(1, -1)), axis=0
            )

        # Group-major reorganization
        # [omega×5, gravity×5, cmd×5, pos×5, vel×5, action×5]
        obs_arr = self.current_obs_history  # (5, 96)
        n = self.config.num_actions  # 29
        obs_input = np.concatenate([
            obs_arr[:, 0:3].reshape(-1),          # omega × 5 frames
            obs_arr[:, 3:6].reshape(-1),          # gravity × 5 frames
            obs_arr[:, 6:9].reshape(-1),          # cmd × 5 frames
            obs_arr[:, 9:9+n].reshape(-1),        # joint pos × 5 frames
            obs_arr[:, 9+n:9+2*n].reshape(-1),    # joint vel × 5 frames
            obs_arr[:, 9+2*n:9+3*n].reshape(-1),  # last action × 5 frames
        ])
        obs_batch = obs_input[np.newaxis, :].astype(np.float32)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_batch)
            action_tensor = self.policy(obs_tensor)
            if isinstance(action_tensor, tuple):
                action_tensor = action_tensor[0]
            self.action = action_tensor.cpu().numpy().flatten().astype(np.float32)

        target_dof_pos = self.config.default_joint_pos + self.action * self.config.action_scale
        with self.cmd_lock:
            for i in range(len(self.config.sdk2isaac_idx)):
                self.low_cmd.motor_cmd[self.config.sdk2isaac_idx[i]].q = target_dof_pos[i]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="enp108s0", help="network interface")
    parser.add_argument("--config_path", type=str, default="config/g1_walk.yaml", help="configuration file path")
    args = parser.parse_args()

    config = Config(args.config_path)
    controller = Controller(config, args.net)

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
