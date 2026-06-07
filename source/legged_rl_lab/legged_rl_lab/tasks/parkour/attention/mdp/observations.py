from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def elevation_map(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    noise: bool = False,
    height_clip: tuple[float, float] = (-1.2, 0.0),
    height_noise_std: float = 0.03,
    height_offset_range: tuple[float, float] = (-0.05, 0.05),
) -> torch.Tensor:
    """Return ray-hit coordinates in the sensor/yaw frame as a flattened AME map."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w
    num_envs, num_rays, _ = relative_pos_w.shape

    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat = sensor_quat.unsqueeze(1).expand(num_envs, num_rays, 4).reshape(num_envs * num_rays, 4)
    sensor_coords = quat_apply_inverse(sensor_quat.to(torch.float), relative_pos_w.reshape(num_envs * num_rays, 3))
    sensor_coords = sensor_coords.reshape(num_envs, num_rays, 3)
    sensor_coords = torch.nan_to_num(sensor_coords)

    if noise:
        if getattr(env, "_attention_map_height_offset", None) is None or env._attention_map_height_offset.shape != (
            num_envs,
            1,
        ):
            env._attention_map_height_offset = torch.zeros((num_envs, 1), device=env.device)

        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_env_ids.numel() > 0:
                lo, hi = height_offset_range
                env._attention_map_height_offset[reset_env_ids] = (
                    torch.rand((reset_env_ids.numel(), 1), device=env.device) * (hi - lo) + lo
                )

        sensor_coords[..., 2] += torch.randn_like(sensor_coords[..., 2]) * height_noise_std
        sensor_coords[..., 2] += env._attention_map_height_offset

    sensor_coords[..., 2] = sensor_coords[..., 2].clamp(min=height_clip[0], max=height_clip[1])
    return sensor_coords.reshape(num_envs, num_rays * 3)
