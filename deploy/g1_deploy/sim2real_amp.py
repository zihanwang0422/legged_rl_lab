#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

from common.command_helper import create_damping_cmd
from common.remote_controller import KeyMap
from config import Config
from sim2real_walk import Controller
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


def main() -> None:
    parser = argparse.ArgumentParser(description="SDK2 sim2real controller for G1 AMP velocity policies.")
    parser.add_argument("--net", type=str, default="enp108s0", help="network interface")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain id, use 1 for local SDK2 MuJoCo bridge")
    parser.add_argument("--config_path", type=str, default="config/g1_amp.yaml", help="configuration file path")
    parser.add_argument("--model", type=str, default="g1_walk.onnx", help="ONNX model filename or path")
    parser.add_argument("--debug_policy", action="store_true", help="Print policy observation/action ranges.")
    args = parser.parse_args()

    config = Config(resolve_config(args.config_path))
    config.set_policy_path(str(resolve_model(args.model)))
    controller = Controller(config, args.net, args.domain_id, debug_policy=args.debug_policy)

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
