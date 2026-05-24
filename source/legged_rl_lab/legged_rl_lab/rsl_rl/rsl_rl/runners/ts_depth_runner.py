from __future__ import annotations
import copy
import os
import time
import warnings
import torch
from tensordict import TensorDict
from rsl_rl.algorithms.ppo_ts_depth import PPO_TSDepth
from rsl_rl.env import VecEnv
from rsl_rl.modules.actor_critic_ts_depth import ActorCriticTSDepth
from rsl_rl.modules.actor_critic_ts_depth_teacher import ActorCriticTSDepthTeacher
from rsl_rl.storage.rollout_storage_ts_depth import RolloutStorageTSDepth
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups
_PPO_TSDEPTH_ALG_KEYS = frozenset({'num_learning_epochs', 'num_mini_batches', 'clip_param', 'gamma', 'lam', 'value_loss_coef', 'entropy_coef', 'learning_rate', 'max_grad_norm', 'use_clipped_value_loss', 'schedule', 'desired_kl', 'use_spo', 'encoder_lr', 'distillation'})

def load_teacher_weights_from_ts_depth_checkpoint(teacher: torch.nn.Module, path: str, map_location: str | torch.device) -> None:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    sd = payload.get('model_state_dict', payload)
    filtered = {k: v for (k, v) in sd.items() if k == 'std' or k.startswith(('privilege_encoder.', 'actor.', 'critic.'))}
    teacher.load_state_dict(filtered, strict=False)

def _algorithm_kwargs_for_ppo_ts_depth(alg_cfg: dict) -> dict:
    return {k: v for (k, v) in alg_cfg.items() if k in _PPO_TSDEPTH_ALG_KEYS}

def _nan_tripwire(env, obs_td, rewards: torch.Tensor, dones: torch.Tensor, prev_actions: torch.Tensor, iteration: int, step_idx: int) -> None:
    issues: list[tuple[str, torch.Tensor]] = []

    def _flag(name: str, tensor, nan_only: bool=False) -> None:
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return
        bad_mask = torch.isnan(tensor) if nan_only else ~torch.isfinite(tensor)
        if not bool(bad_mask.any()):
            return
        flat = bad_mask.reshape(bad_mask.shape[0], -1).any(dim=-1)
        bad_envs = flat.nonzero(as_tuple=False).flatten()
        if bad_envs.numel() > 0:
            issues.append((name, bad_envs))
    try:
        robot = env.scene['robot']
        rd = robot.data
        _flag('robot.data.root_pos_w', rd.root_pos_w)
        _flag('robot.data.root_quat_w', rd.root_quat_w)
        _flag('robot.data.joint_pos', rd.joint_pos)
        _flag('robot.data.joint_vel', rd.joint_vel)
        if hasattr(rd, 'body_lin_vel_w'):
            _flag('robot.data.body_lin_vel_w', rd.body_lin_vel_w)
        if hasattr(rd, 'body_ang_vel_w'):
            _flag('robot.data.body_ang_vel_w', rd.body_ang_vel_w)
    except Exception as e:
        print(f'[nan-tripwire] could not read robot state: {e}')
    for (sname, sensor) in env.scene.sensors.items():
        sd = getattr(sensor, 'data', None)
        if sd is None:
            continue
        _flag(f'sensor[{sname}].pos_w', getattr(sd, 'pos_w', None))
        _flag(f'sensor[{sname}].quat_w', getattr(sd, 'quat_w', None))
        _flag(f'sensor[{sname}].ray_hits_w (NaN-only)', getattr(sd, 'ray_hits_w', None), nan_only=True)
        _flag(f'sensor[{sname}].net_forces_w', getattr(sd, 'net_forces_w', None))
        _flag(f'sensor[{sname}].net_forces_w_history', getattr(sd, 'net_forces_w_history', None))
    if isinstance(obs_td, TensorDict):
        for key in obs_td.keys():
            v = obs_td[key]
            if isinstance(v, torch.Tensor):
                _flag(f'obs[{key}]', v)
    elif isinstance(obs_td, torch.Tensor):
        _flag('obs', obs_td)
    _flag('rewards', rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards)
    if not issues:
        return
    print('\n' + '=' * 78)
    print(f'[NAN-TRIPWIRE] iteration={iteration}, step_idx={step_idx}')
    print('=' * 78)
    print('Detected NaN/Inf sources (ordered: physics → sensors → obs → rewards):')
    for (name, bad_envs) in issues:
        envs_preview = bad_envs[:8].tolist()
        more = '' if bad_envs.numel() <= 8 else f' ... (+{bad_envs.numel() - 8} more)'
        print(f'  [{len(bad_envs):4d} envs] {name}: {envs_preview}{more}')
    (first_name, first_bad) = issues[0]
    sample = first_bad[:3]
    print(f'\nContext dump for envs {sample.tolist()} (the lowest-layer NaN source: {first_name}):')
    try:
        robot = env.scene['robot']
        rd = robot.data
        prev_act_sample = prev_actions[sample].detach().cpu()
        print(f'  prev_action          : max|.|={prev_act_sample.abs().amax(dim=-1).tolist()}')
        print(f'                          values={prev_act_sample.tolist()}')
        quat = rd.root_quat_w[sample].detach().cpu()
        print(f'  root_quat_w          : {quat.tolist()}')
        print(f'  root_quat_norm       : {quat.norm(dim=-1).tolist()}  (should be ~1.0)')
        pos = rd.root_pos_w[sample].detach().cpu()
        print(f'  root_pos_w           : {pos.tolist()}')
        jp = rd.joint_pos[sample].detach().cpu()
        print(f'  joint_pos max|.|     : {jp.abs().amax(dim=-1).tolist()}')
        jv = rd.joint_vel[sample].detach().cpu()
        print(f'  joint_vel max|.|     : {jv.abs().amax(dim=-1).tolist()}')
    except Exception as e:
        print(f'  (could not dump robot context: {e})')
    try:
        cf = env.scene.sensors.get('contact_sensor', None)
        if cf is not None and hasattr(cf.data, 'net_forces_w'):
            f = cf.data.net_forces_w[sample].detach().cpu()
            print(f'  contact_forces max|.|: {f.reshape(f.shape[0], -1).abs().amax(dim=-1).tolist()}')
    except Exception as e:
        print(f'  (could not dump contact forces: {e})')
    print(f'\n  done flag for sample : {dones[sample].tolist()}')
    print(f'  reward for sample    : {(rewards[sample].tolist() if rewards.dim() == 1 else rewards[sample].tolist())}')
    print('=' * 78 + '\n')
    raise RuntimeError(f'[NAN-TRIPWIRE] First NaN source: {first_name} on {len(first_bad)} envs at iteration={iteration}, step_idx={step_idx}. See diagnostic dump above.')

class TsDepthRunner(OnPolicyRunner):

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None=None, device: str='cpu') -> None:
        train_cfg['algorithm'].setdefault('rnd_cfg', None)
        self.cfg = train_cfg
        self.policy_cfg_template = copy.deepcopy(train_cfg['policy'])
        self.alg_cfg_template = copy.deepcopy(train_cfg['algorithm'])
        self.policy_cfg = train_cfg['policy']
        self.alg_cfg = train_cfg['algorithm']
        self.device = device
        self.env = env
        self._configure_multi_gpu()
        obs = self.env.get_observations()
        self.cfg['obs_groups'] = resolve_obs_groups(obs, self.cfg['obs_groups'], self._get_default_obs_sets())
        ds = tuple(self.policy_cfg_template['depth_shape'])
        (self._depth_c, self._depth_h, self._depth_w) = ds
        self._depth_flat_dim = self._depth_c * self._depth_h * self._depth_w
        self.alg = self._construct_ts_depth_algorithm(obs)
        from rsl_rl.utils.logger import Logger
        self.logger = Logger(log_dir=log_dir, cfg=self.cfg, env_cfg=self.env.cfg, num_envs=self.env.num_envs, is_distributed=self.is_distributed, gpu_world_size=self.gpu_world_size, gpu_global_rank=self.gpu_global_rank, device=self.device)
        self.current_learning_iteration = 0

    def _get_default_obs_sets(self) -> list[str]:
        return ['policy', 'privileged', 'depth', 'critic']

    def _flat_from_td(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        og = self.cfg['obs_groups']
        actor = torch.cat([obs[k] for k in og['policy']], dim=-1)
        priv = torch.cat([obs[k] for k in og['privileged']], dim=-1)
        critic = torch.cat([obs[k] for k in og['critic']], dim=-1)
        depth_flat = torch.cat([obs[k] for k in og['depth']], dim=-1)
        if depth_flat.shape[-1] != self._depth_flat_dim:
            raise ValueError(f"Depth flat dim {depth_flat.shape[-1]} != C*H*W={self._depth_flat_dim} from depth_shape={self.policy_cfg_template['depth_shape']}")
        depth_img = depth_flat.view(depth_flat.shape[0], self._depth_c, self._depth_h, self._depth_w)
        return (actor, priv, depth_img, critic)

    def _construct_ts_depth_algorithm(self, obs: TensorDict) -> PPO_TSDepth:
        if self.cfg.get('empirical_normalization') is not None:
            warnings.warn('TsDepthRunner does not support empirical_normalization; ignoring.', UserWarning)
        policy_cfg = copy.deepcopy(self.policy_cfg_template)
        alg_cfg = copy.deepcopy(self.alg_cfg_template)
        alg_cfg.pop('rnd_cfg', None)
        teacher_ckpt = (alg_cfg.pop('teacher_checkpoint_path', None) or '').strip()
        class_name = policy_cfg.pop('class_name')
        if class_name != 'ActorCriticTSDepth':
            raise ValueError(f"TsDepthRunner expects policy.class_name 'ActorCriticTSDepth', got {class_name!r}")
        (actor, priv, depth_img, critic) = self._flat_from_td(obs)
        num_actor_obs = actor.shape[-1]
        num_priv_obs = priv.shape[-1]
        num_critic_obs = critic.shape[-1]
        num_student = policy_cfg.pop('num_student_envs', self.env.num_envs)
        if num_student is None:
            num_student = self.env.num_envs
        num_student = int(num_student)
        num_student = max(1, min(num_student, self.env.num_envs))
        num_latent_dims = policy_cfg.pop('num_latent_dims')
        policy_cfg.pop('depth_shape', None)
        depth_resolution_hw = (self._depth_h, self._depth_w)
        actor_critic = ActorCriticTSDepth(num_actor_obs, self.env.num_actions, num_priv_obs, num_latent_dims, num_critic_obs, depth_resolution_hw, **policy_cfg).to(self.device)
        distillation = bool(alg_cfg.get('distillation', False))
        storage_num_envs = num_student if distillation else self.env.num_envs
        storage = RolloutStorageTSDepth(storage_num_envs, num_student, self.cfg['num_steps_per_env'], (num_actor_obs,), (num_priv_obs,), (self._depth_c, self._depth_h, self._depth_w), (num_critic_obs,), (self.env.num_actions,), self.device)
        alg_class_name = alg_cfg.pop('class_name')
        if alg_class_name != 'PPO_TSDepth':
            raise ValueError(f"TsDepthRunner expects algorithm.class_name 'PPO_TSDepth', got {alg_class_name!r}")
        teacher_module: ActorCriticTSDepthTeacher | None = None
        if distillation:
            if not teacher_ckpt:
                raise ValueError('algorithm.distillation=True requires algorithm.teacher_checkpoint_path (path to a phase-1 TsDepth checkpoint), or train with --resume so the teacher defaults to that file.')
            pt = copy.deepcopy(self.policy_cfg_template)
            teacher_ctor = dict(actor_hidden_dims=pt.get('actor_hidden_dims', [512, 256, 128]), critic_hidden_dims=pt.get('critic_hidden_dims', [512, 256, 128]), privilege_encoder_hidden_dims=pt.get('privilege_encoder_hidden_dims', [256, 128]), activation=pt.get('activation', 'elu'), init_noise_std=pt.get('init_noise_std', 1.0), clip_actions=float(pt.get('clip_actions', 100.0)))
            teacher_module = ActorCriticTSDepthTeacher(num_actor_obs, self.env.num_actions, num_priv_obs, num_latent_dims, num_critic_obs, **teacher_ctor).to(self.device)
            load_teacher_weights_from_ts_depth_checkpoint(teacher_module, teacher_ckpt, self.device)
            teacher_module.eval()
            for p in teacher_module.parameters():
                p.requires_grad_(False)
        alg_kw = _algorithm_kwargs_for_ppo_ts_depth(alg_cfg)
        alg: PPO_TSDepth = PPO_TSDepth(actor_critic, device=self.device, num_student=num_student, teacher_actor_critic=teacher_module, **alg_kw)
        alg.init_storage(storage)
        return alg

    def _construct_algorithm(self, obs: TensorDict):
        raise RuntimeError('TsDepthRunner constructs the algorithm in __init__')

    def train_mode(self) -> None:
        self.alg.actor_critic.train()
        if self.alg.teacher_actor_critic is not None:
            self.alg.teacher_actor_critic.eval()

    def eval_mode(self) -> None:
        self.alg.actor_critic.eval()
        if self.alg.teacher_actor_critic is not None:
            self.alg.teacher_actor_critic.eval()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool=False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs_td = self.env.get_observations().to(self.device)
        self.train_mode()
        if self.is_distributed:
            print(f'Synchronizing parameters for rank {self.gpu_global_rank}...')
        self.logger.init_logging_writer()
        distill_slice = self.alg.distillation and self.alg.num_student < self.env.num_envs
        num_student = int(self.alg.num_student)
        num_envs = int(self.env.num_envs)
        num_actions = int(self.env.num_actions)
        if distill_slice and self.logger.num_envs != num_student:
            self.logger.num_envs = num_student
            self.logger.cur_reward_sum = torch.zeros(num_student, dtype=torch.float, device=self.device)
            self.logger.cur_episode_length = torch.zeros(num_student, dtype=torch.float, device=self.device)
            if hasattr(self.logger, 'cur_ereward_sum'):
                self.logger.cur_ereward_sum = torch.zeros(num_student, dtype=torch.float, device=self.device)
                self.logger.cur_ireward_sum = torch.zeros(num_student, dtype=torch.float, device=self.device)
        nan_tripwire_enabled = bool(self.cfg.get('nan_tripwire', True))
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for step_idx in range(self.cfg['num_steps_per_env']):
                    (actor, priv, depth_img, critic) = self._flat_from_td(obs_td)
                    if distill_slice:
                        actor_s = actor[:num_student]
                        priv_s = priv[:num_student]
                        depth_s = depth_img[:num_student]
                        critic_s = critic[:num_student]
                        actions_s = self.alg.act(actor_s, priv_s, depth_s, critic_s)
                        actions = torch.zeros(num_envs, num_actions, device=actions_s.device)
                        actions[:num_student] = actions_s
                    else:
                        actions = self.alg.act(actor, priv, depth_img, critic)
                    (obs_td, rewards, dones, extras) = self.env.step(actions.to(self.env.device))
                    obs_td = obs_td.to(self.device)
                    (rewards, dones) = (rewards.to(self.device), dones.to(self.device))
                    if nan_tripwire_enabled:
                        _nan_tripwire(self.env.unwrapped, obs_td, rewards, dones, actions, iteration=it, step_idx=step_idx)
                    if distill_slice:
                        rewards = rewards[:num_student]
                        dones = dones[:num_student]
                    self.alg.process_env_step(rewards, dones, extras)
                    intrinsic_rewards = None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                collect_time = time.time() - start
                start = time.time()
                (_, _, _, last_critic) = self._flat_from_td(obs_td)
                if distill_slice:
                    last_critic = last_critic[:num_student]
                self.alg.compute_returns(last_critic)
            (mean_value_loss, mean_surrogate_loss, mean_latent_loss, mean_action_rec_loss) = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it
            loss_dict = {'value_function': float(mean_value_loss), 'surrogate': float(mean_surrogate_loss), 'latent_reconstruction': float(mean_latent_loss), 'action_reconstruction': float(mean_action_rec_loss)}
            self.logger.log(it=it, start_it=start_it, total_it=total_it, collect_time=collect_time, learn_time=learn_time, loss_dict=loss_dict, learning_rate=self.alg.learning_rate, action_std=self.alg.actor_critic.std, rnd_weight=None)
            if it % self.cfg['save_interval'] == 0:
                self.save(os.path.join(self.logger.log_dir, f'model_{it}.pt'))
            if it % 5 == 0:
                self.alg.actor_critic.detach_hidden_states()
        if self.logger.log_dir is not None and (not self.logger.disable_logs):
            self.save(os.path.join(self.logger.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def save(self, path: str, infos: dict | None=None) -> None:
        saved_dict = {'model_state_dict': self.alg.actor_critic.state_dict(), 'teacher_optimizer_state_dict': self.alg.teacher_optimizer.state_dict(), 'student_optimizer_state_dict': self.alg.student_optimizer.state_dict(), 'iter': self.current_learning_iteration, 'infos': infos}
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool=True, map_location: str | None=None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            if 'teacher_optimizer_state_dict' in loaded_dict:
                try:
                    self.alg.teacher_optimizer.load_state_dict(loaded_dict['teacher_optimizer_state_dict'])
                except (ValueError, KeyError) as e:
                    print(f'[WARNING][TsDepthRunner.load] teacher_optimizer state mismatch, starting teacher optimizer fresh. Details: {e}')
            if 'student_optimizer_state_dict' in loaded_dict:
                try:
                    self.alg.student_optimizer.load_state_dict(loaded_dict['student_optimizer_state_dict'])
                except (ValueError, KeyError) as e:
                    print(f'[WARNING][TsDepthRunner.load] student_optimizer state mismatch (usually phase 1 → phase 2 transition), starting student optimizer fresh. Details: {e}')
        self.current_learning_iteration = loaded_dict.get('iter', 0)
        return loaded_dict.get('infos', {})

    def get_inference_policy(self, device: str | None=None) -> callable:
        self.eval_mode()
        dev = device or self.device
        ac = self.alg.actor_critic
        if device is not None:
            ac.to(device)

        def policy(obs_td: TensorDict) -> torch.Tensor:
            obs_td = obs_td.to(dev)
            (actor, _, depth_img, _) = self._flat_from_td(obs_td)
            with torch.inference_mode():
                return ac.act_inference(actor, depth_img)
        return policy
