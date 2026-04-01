import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from legged_rl_lab.tasks.tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu",
    ):
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos=None):
        """Save the model and training information, and auto-export ONNX."""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        policy_nn = self.alg.policy
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None
        export_motion_policy_as_onnx(
            self.env.unwrapped, policy_nn, normalizer=normalizer, path=policy_path, filename=filename
        )
