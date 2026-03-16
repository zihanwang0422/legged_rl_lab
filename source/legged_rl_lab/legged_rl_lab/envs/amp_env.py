# Copyright (c) 2024-2025 zihan wang
# SPDX-License-Identifier: Apache-2.0

"""AMP ManagerBasedRLEnv.

This class inherits from the ``ManagerBasedRLEnv`` class and is used to create an environment
for training and testing reinforcement learning agents on locomotion tasks using AMP
(Adversarial Motion Priors).

In the original ``ManagerBasedRLEnv``'s ``step`` method, observations are computed **after**
environments are reset. But for AMP, we need to record the AMP observations **before** the reset
occurs, so the discriminator can evaluate the (pre_amp_obs, post_amp_obs) transition pair on the
actual physics transition — not on artificial post-reset states.

This class overrides the ``step`` method to:

1. Compute AMP observations **before** the auto-reset.
2. Store them in ``extras["amp_obs"]`` so the training script can access them.
3. Then proceed with the normal reset and post-reset observation computation.

The design is robot-agnostic — it works with any robot that has an ``amp`` observation group
configured in its env config.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class AMPManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv extended for AMP training.

    Key difference from the base class: AMP observations are computed **before** the reset
    and stored in ``extras["amp_obs"]``, so the AMP (state, next_state) transition pairs
    reflect the actual physics transition rather than a post-reset artifact.

    The ``amp_obs_group`` parameter specifies which observation group to treat as the AMP
    observation (default: ``"amp"``).
    """

    cfg: ManagerBasedRLEnvCfg
    """The configuration object for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._amp_obs_group = "amp"

    def step(self, action: torch.Tensor):
        """Override step to capture AMP observations before reset.

        Follows the exact flow of ``ManagerBasedRLEnv.step()`` in IsaacLab 2.3.0, with the
        single addition of computing AMP observations after rewards/terminations but before
        environments are reset.

        Flow:
        1. Process actions and record pre-step
        2. Physics decimation loop (apply_action → write → sim.step → render → update)
        3. Post-step counters + terminations + rewards
        4. Recorder post-step (conditional)
        5. **[AMP] Compute AMP obs before reset** → store in ``extras["amp_obs"]``
        6. Reset terminated envs (with recorder pre/post-reset calls)
        7. Commands + interval events
        8. Final observations (post-reset, with history update)
        9. Return
        """
        # -- 1. Process actions
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        # -- 2. Physics decimation loop (matches ManagerBasedRLEnv exactly)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # apply actions into actuators
            self.action_manager.apply_action()
            # write actions to sim
            self.scene.write_data_to_sim()
            # simulate one physics step
            self.sim.step(render=False)
            # recorder hook for each decimation step
            self.recorder_manager.record_post_physics_decimation_step()
            # render at configured interval
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update scene buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # -- 3. Post-step counters, terminations, rewards
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- 4. Recorder post-step (only when recorder terms are active)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- 5. [AMP SPECIFIC] Capture AMP observations BEFORE any env reset.
        # This ensures the discriminator sees genuine physics transitions, not
        # post-reset states for terminated environments.
        # Use `active_terms` (IsaacLab 2.x) with fallback to `_group_obs_term_names`.
        _obs_groups = getattr(
            self.observation_manager, "active_terms",
            getattr(self.observation_manager, "_group_obs_term_names", {})
        )
        if self._amp_obs_group in _obs_groups:
            self.extras["amp_obs"] = self.observation_manager.compute_group(
                self._amp_obs_group, update_history=False
            )

        # -- 6. Reset terminated environments
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            # Re-render after reset if RTX sensors are present
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- 7. Commands and interval events
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # -- 8. Final observations (post-reset, history updated for all groups)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # -- 9. Add time_outs for reward bootstrapping in RL algorithms
        self.extras["time_outs"] = self.reset_time_outs

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
