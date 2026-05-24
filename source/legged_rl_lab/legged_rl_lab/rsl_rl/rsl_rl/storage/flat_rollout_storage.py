from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Tuple, List, Generator, Union
from rsl_rl.utils import split_and_pad_trajectories

class FlatRolloutStorage:

    class Transition:

        def __init__(self) -> None:
            self.observations: Optional[torch.Tensor] = None
            self.critic_observations: Optional[torch.Tensor] = None
            self.actions: Optional[torch.Tensor] = None
            self.rewards: Optional[torch.Tensor] = None
            self.dones: Optional[torch.Tensor] = None
            self.values: Optional[torch.Tensor] = None
            self.actions_log_prob: Optional[torch.Tensor] = None
            self.action_mean: Optional[torch.Tensor] = None
            self.action_sigma: Optional[torch.Tensor] = None
            self.hidden_states: Optional[Tuple[torch.Tensor, ...]] = None

        def clear(self) -> None:
            self.__init__()

    def __init__(self, num_envs: int, num_transitions_per_env: int, obs_shape: Tuple[int, ...], privileged_obs_shape: Tuple[Optional[int], ...], actions_shape: Tuple[int, ...], device: str='cpu') -> None:
        self.device: str = device
        self.obs_shape: Tuple[int, ...] = obs_shape
        self.privileged_obs_shape: Tuple[Optional[int], ...] = privileged_obs_shape
        self.actions_shape: Tuple[int, ...] = actions_shape
        self.observations: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.privileged_observations: Optional[torch.Tensor]
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.actions_log_prob: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.num_transitions_per_env: int = num_transitions_per_env
        self.num_envs: int = num_envs
        self.saved_hidden_states_a: Optional[List[torch.Tensor]] = None
        self.saved_hidden_states_c: Optional[List[torch.Tensor]] = None
        self.step: int = 0

    def add_transitions(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise AssertionError('Rollout buffer overflow')
        assert transition.observations is not None
        assert transition.actions is not None
        assert transition.rewards is not None
        assert transition.dones is not None
        assert transition.values is not None
        assert transition.actions_log_prob is not None
        assert transition.action_mean is not None
        assert transition.action_sigma is not None
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None and transition.critic_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states: Optional[Tuple[torch.Tensor, ...]]) -> None:
        if hidden_states is None or hidden_states == (None, None):
            return
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        assert self.saved_hidden_states_a is not None
        assert self.saved_hidden_states_c is not None
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self) -> None:
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-08)

    def get_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return (trajectory_lengths.float().mean(), self.rewards.mean())

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int=8) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[None, None], None], None, None]:
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield (obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None)

    def reccurent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int=8) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor], None, None]:
        (padded_obs_trajectories, trajectory_masks) = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            (padded_critic_obs_trajectories, _) = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories
        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                last_was_done = last_was_done.permute(1, 0)
                assert self.saved_hidden_states_a is not None
                assert self.saved_hidden_states_c is not None
                hid_a_batch = [saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous() for saved_hidden_states in self.saved_hidden_states_a]
                hid_c_batch = [saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous() for saved_hidden_states in self.saved_hidden_states_c]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch
                yield (obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch)
                first_traj = last_traj