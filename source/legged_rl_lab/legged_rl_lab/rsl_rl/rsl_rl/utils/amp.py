from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import torch

from legged_rl_lab.managers import MotionLoader


class RunningMeanStd:
    """Running mean / variance tracker with numpy storage."""

    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class Normalizer(RunningMeanStd):
    """Feature normalizer compatible with TienKung-style AMP."""

    def __init__(self, input_dim: int, epsilon: float = 1e-4, clip_obs: float = 10.0) -> None:
        super().__init__(shape=(input_dim,))
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize_torch(self, value: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((value - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)


class AMPLoader:
    """Clip-aware AMP expert transition loader.

    The loader preserves per-motion clip boundaries so sampled expert
    transitions never cross from the end of one clip into the start of another.
    Each trajectory is stored as a tensor of per-frame AMP features with layout
    matching the environment's single-frame ``amp`` observation group.
    """

    SUPPORTED_EXTS = (".npy", ".npz", ".pt", ".pth", ".csv")

    def __init__(
        self,
        device: str | torch.device,
        time_between_frames: float,
        motion_files: str | os.PathLike | Iterable[str | os.PathLike],
        robot: str = "g1",
        preload_transitions: bool = False,
        num_preload_transitions: int = 1_000_000,
    ) -> None:
        self.device = device
        self.time_between_frames = time_between_frames
        self.robot = robot
        self.preload_transitions = preload_transitions

        self.trajectories: list[torch.Tensor] = []
        self.trajectory_names: list[str] = []
        self.trajectory_idxs: list[int] = []
        self.trajectory_num_frames: np.ndarray | None = None
        self.trajectory_weights: np.ndarray | None = None

        motion_paths = self._resolve_motion_paths(motion_files)
        if not motion_paths:
            raise FileNotFoundError(f"No motion files found from: {motion_files}")

        for motion_path in motion_paths:
            loader = MotionLoader(device=str(device), robot=robot)
            try:
                trajectory = loader.load(motion_path).float().to(device)
            except Exception:
                continue
            if trajectory.shape[0] < 2:
                continue
            self.trajectories.append(trajectory)
            self.trajectory_names.append(os.path.basename(motion_path))
            self.trajectory_idxs.append(len(self.trajectory_idxs))

        if not self.trajectories:
            raise RuntimeError(f"Failed to load any valid AMP trajectories from: {motion_files}")

        self.trajectory_num_frames = np.array([traj.shape[0] for traj in self.trajectories], dtype=np.int64)
        valid_transition_counts = np.maximum(self.trajectory_num_frames - 1, 1)
        self.trajectory_weights = valid_transition_counts.astype(np.float64)
        self.trajectory_weights /= np.sum(self.trajectory_weights)

        if self.preload_transitions:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            frame_idxs = self.frame_idx_sample_batch(traj_idxs)
            self.preloaded_s, self.preloaded_s_next = self.get_transition_batch(traj_idxs, frame_idxs)

    def _resolve_motion_paths(self, motion_files: str | os.PathLike | Iterable[str | os.PathLike]) -> list[str]:
        if isinstance(motion_files, (str, os.PathLike)):
            items = [motion_files]
        else:
            items = list(motion_files)

        resolved: list[str] = []
        for item in items:
            path = os.fspath(item)
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for filename in sorted(files):
                        if filename.lower().endswith(self.SUPPORTED_EXTS):
                            resolved.append(os.path.join(root, filename))
            elif os.path.isfile(path):
                resolved.append(path)
        return resolved

    def weighted_traj_idx_sample_batch(self, size: int) -> np.ndarray:
        assert self.trajectory_weights is not None
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def frame_idx_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
        assert self.trajectory_num_frames is not None
        frame_idxs = np.empty(len(traj_idxs), dtype=np.int64)
        for i, traj_idx in enumerate(traj_idxs):
            max_start = int(self.trajectory_num_frames[traj_idx] - 1)
            frame_idxs[i] = np.random.randint(0, max_start)
        return frame_idxs

    def get_transition_batch(self, traj_idxs: np.ndarray, frame_idxs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        states = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        next_states = torch.zeros_like(states)
        for traj_idx in set(traj_idxs.tolist()):
            mask = traj_idxs == traj_idx
            idx = torch.as_tensor(frame_idxs[mask], device=self.device, dtype=torch.long)
            traj = self.trajectories[traj_idx]
            states[mask] = traj[idx]
            next_states[mask] = traj[idx + 1]
        return states, next_states

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                yield self.preloaded_s[idxs], self.preloaded_s_next[idxs]
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                frame_idxs = self.frame_idx_sample_batch(traj_idxs)
                yield self.get_transition_batch(traj_idxs, frame_idxs)

    @property
    def observation_dim(self) -> int:
        return int(self.trajectories[0].shape[1])
