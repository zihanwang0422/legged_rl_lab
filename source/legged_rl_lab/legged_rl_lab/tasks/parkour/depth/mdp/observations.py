from __future__ import annotations
import math
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse

def height_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float=0.5, clip: tuple[float, float] | None=None) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    h = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    if clip is not None:
        h = h.clamp(min=clip[0], max=clip[1])
    return h

def depth_image_camera(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_depth: float=3.0, data_type: str='distance_to_image_plane') -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output[data_type]
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=max_depth)
    depth = depth.clamp(min=0.0, max=max_depth) / max_depth
    return depth.flatten(start_dim=1)

def gait_phase_sin_cos(env: ManagerBasedRLEnv, period: float=0.8, offset: list[float] | tuple[float, ...]=(0.0, 0.5)) -> torch.Tensor:
    t = env.episode_length_buf.float() * env.step_dt
    global_phase = t % period / period
    phases = torch.stack([(global_phase + off) % 1.0 for off in offset], dim=-1)
    sin_phase = torch.sin(2 * math.pi * phases)
    cos_phase = torch.cos(2 * math.pi * phases)
    return torch.cat([sin_phase, cos_phase], dim=-1)

def feet_pos_body_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel = feet_pos_w - root_pos.unsqueeze(1)
    num_feet = rel.shape[1]
    parts = [quat_apply_inverse(root_quat, rel[:, i]) for i in range(num_feet)]
    return torch.cat(parts, dim=-1)

def feet_vel_body_frame(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_vel = asset.data.root_lin_vel_w
    root_quat = asset.data.root_quat_w
    feet_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    rel = feet_vel_w - root_vel.unsqueeze(1)
    num_feet = rel.shape[1]
    parts = [quat_apply_inverse(root_quat, rel[:, i]) for i in range(num_feet)]
    return torch.cat(parts, dim=-1)

def feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return forces.reshape(forces.shape[0], -1)

def scalar_rigid_friction_mean(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    mats = asset.root_physx_view.get_material_properties().to(env.device)
    mu_s = mats[:, :, 0].mean(dim=1)
    mu_d = mats[:, :, 1].mean(dim=1)
    return ((mu_s + mu_d) * 0.5).unsqueeze(-1)

def body_mass_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    m = asset.root_physx_view.get_masses().to(env.device)
    d = asset.data.default_mass.to(env.device)
    ratio = m[:, asset_cfg.body_ids] / (d[:, asset_cfg.body_ids] + 1e-08)
    return ratio.mean(dim=-1, keepdim=True)

def last_push_delta_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, '_ts_depth_push_xy'):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env._ts_depth_push_xy.to(device=env.device)

def joint_stiffness_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot', joint_names=['.*'])) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    s = asset.data.joint_stiffness
    d = asset.data.default_joint_stiffness
    if s is None or d is None:
        return torch.ones(env.num_envs, asset.num_joints, device=env.device)
    s = s[:, asset_cfg.joint_ids]
    d = d[:, asset_cfg.joint_ids]
    return s / (d + 1e-08)

def joint_damping_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot', joint_names=['.*'])) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    damp = asset.data.joint_damping
    d0 = asset.data.default_joint_damping
    if damp is None or d0 is None:
        return torch.ones(env.num_envs, asset.num_joints, device=env.device)
    damp = damp[:, asset_cfg.joint_ids]
    d0 = d0[:, asset_cfg.joint_ids]
    return damp / (d0 + 1e-08)

def links_contact_binary(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float=1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact.float()

def height_relative_to_feet(env: ManagerBasedRLEnv, sensor_names: list[str], asset_cfg: SceneEntityCfg=SceneEntityCfg('robot'), clip: tuple[float, float] | None=(-1.0, 1.0)) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    parts = []
    for (i, name) in enumerate(sensor_names):
        sensor: RayCaster = env.scene.sensors[name]
        hits_z = sensor.data.ray_hits_w[..., 2]
        rel = foot_z[:, i:i + 1] - hits_z
        if clip is not None:
            rel = rel.clamp(min=clip[0], max=clip[1])
        parts.append(rel)
    return torch.cat(parts, dim=-1)

def normal_vector_around_feet(env: ManagerBasedRLEnv, sensor_names: list[str]) -> torch.Tensor:
    parts = []
    for name in sensor_names:
        sensor: RayCaster = env.scene.sensors[name]
        hits = sensor.data.ray_hits_w
        p0 = hits[:, 0]
        p1 = hits[:, 2]
        p2 = hits[:, 6]
        valid = torch.isfinite(p0).all(dim=-1) & torch.isfinite(p1).all(dim=-1) & torch.isfinite(p2).all(dim=-1)
        v1 = p1 - p0
        v2 = p2 - p0
        normal = torch.cross(v1, v2, dim=-1)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True).clamp_min(1e-06)
        default = torch.zeros_like(normal)
        default[..., 2] = 1.0
        normal = torch.where(valid.unsqueeze(-1), normal, default)
        parts.append(normal)
    return torch.cat(parts, dim=-1)

def body_com_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    com_b = asset.data.body_com_pos_b[:, asset_cfg.body_ids, :]
    return com_b.reshape(com_b.shape[0], -1)