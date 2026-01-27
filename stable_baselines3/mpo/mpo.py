"""
MPO-style implementation for Stable-Baselines3.

This file implements a *Maximum a Posteriori Policy Optimisation (MPO)*-like update:
  - E-step: sample actions from a target policy and compute nonparametric weights via a
    temperature-constrained softmax over Q-values.
  - M-step: update the online Gaussian policy by minimizing a weighted cross-entropy subject to
    KL constraints (mean/stddev), enforced via dual variables (alphas).

Integration note:
  - Rollouts are collected via SB3's OnPolicyAlgorithm, then converted into an off-policy replay
    buffer that MPO-style updates sample from.
Limitations:
  - Continuous Box observation/action spaces only. Discrete environments are not supported.
"""

from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

import copy
import math
from torch import nn
import torch.distributions as torch_dist

from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule

SelfMPO = TypeVar("SelfMPO", bound="MPO")

_MPO_FLOAT_EPSILON = 1e-8


def _as_independent_normal(d: Any) -> torch_dist.Independent:
    """
    Best-effort adapter to get a torch Independent(Normal) from SB3 distributions.
    Supports:
      - SB3 Distribution objects with `.distribution`,
      - torch Distribution directly.
    """
    base = getattr(d, "distribution", d)
    if isinstance(base, torch_dist.Independent):
        return base
    if isinstance(base, torch_dist.Normal):
        return torch_dist.Independent(base, 1)
    # Some SB3 dists may wrap a Normal but not as Independent.
    if hasattr(base, "base_dist") and isinstance(base.base_dist, torch_dist.Normal):
        return torch_dist.Independent(base.base_dist, 1)
    raise TypeError(f"Unsupported distribution type for MPO: {type(d)}")


def _diag_normal_kl(
    p_mean: th.Tensor,
    p_std: th.Tensor,
    q_mean: th.Tensor,
    q_std: th.Tensor,
    per_dim_constraining: bool,
) -> th.Tensor:
    var_p = p_std.pow(2)
    var_q = q_std.pow(2)
    diff = q_mean - p_mean
    log_term = th.log(q_std / p_std)
    frac_term = (var_p + diff.pow(2)) / (2.0 * var_q)
    kl_elem = log_term + frac_term - 0.5
    return kl_elem if per_dim_constraining else kl_elem.sum(dim=-1)


def _weights_and_temperature_loss(
    q_values: th.Tensor,  # [N,B]
    epsilon: float,
    temperature: th.Tensor,  # [1]
) -> tuple[th.Tensor, th.Tensor]:
    tempered = q_values.detach() / temperature
    weights = th.softmax(tempered, dim=0).detach()
    q_lse = th.logsumexp(tempered, dim=0)  # [B]
    log_num_actions = temperature.new_tensor(math.log(float(q_values.shape[0])))
    eps_t = temperature.new_tensor(float(epsilon))
    loss_temperature = temperature * (eps_t + q_lse.mean() - log_num_actions)
    return weights, loss_temperature


def _nonparametric_kl_from_weights(weights: th.Tensor) -> th.Tensor:
    n = float(weights.shape[0])
    integrand = th.log(n * weights + _MPO_FLOAT_EPSILON)
    return (weights * integrand).sum(dim=0)  # [B]


def _cross_entropy_loss(
    sampled_actions: th.Tensor,  # [N,B,D]
    weights: th.Tensor,  # [N,B]
    action_dist: torch_dist.Distribution,
) -> th.Tensor:
    logp = action_dist.log_prob(sampled_actions)  # [N,B]
    return (-th.sum(logp * weights, dim=0)).mean()


def _parametric_kl_penalty_and_dual(
    kl: th.Tensor,  # [B,D] or [B]
    alpha: th.Tensor,  # [D] or [1]
    epsilon: float,
) -> tuple[th.Tensor, th.Tensor]:
    mean_kl = kl.mean(dim=0)
    loss_kl = (alpha.detach() * mean_kl).sum()
    loss_alpha = (alpha * (float(epsilon) - mean_kl.detach())).sum()
    return loss_kl, loss_alpha


class MPOLoss(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        epsilon: float = 1e-1,
        epsilon_mean: float = 2.5e-3,
        epsilon_stddev: float = 1e-6,
        init_log_temperature: float = 10.0,
        init_log_alpha_mean: float = 10.0,
        init_log_alpha_stddev: float = 1000.0,
        per_dim_constraining: bool = True,
        action_penalization: bool = False,
        epsilon_penalty: float = 1e-3,
    ):
        super().__init__()
        self._epsilon = float(epsilon)
        self._epsilon_mean = float(epsilon_mean)
        self._epsilon_stddev = float(epsilon_stddev)
        self._per_dim = bool(per_dim_constraining)
        self._action_penalization = bool(action_penalization)
        self._epsilon_penalty = float(epsilon_penalty)

        self.log_temperature = nn.Parameter(th.tensor([init_log_temperature], dtype=th.float32))
        alpha_shape = (action_dim,) if self._per_dim else (1,)
        self.log_alpha_mean = nn.Parameter(th.full(alpha_shape, init_log_alpha_mean, dtype=th.float32))
        self.log_alpha_stddev = nn.Parameter(th.full(alpha_shape, init_log_alpha_stddev, dtype=th.float32))

        if self._action_penalization:
            self._log_penalty_temperature = nn.Parameter(th.tensor([init_log_temperature], dtype=th.float32))

        self.register_buffer("min_log_temperature", th.tensor(-18.0, dtype=th.float32))
        self.register_buffer("min_log_alpha", th.tensor(-18.0, dtype=th.float32))

    def forward(
        self,
        *,
        actions: th.Tensor,  # [N,B,D]
        q_values: th.Tensor,  # [N,B]
        online_mean: th.Tensor,  # [B,D]
        online_std: th.Tensor,  # [B,D]
        target_mean: th.Tensor,  # [B,D]
        target_std: th.Tensor,  # [B,D]
    ) -> tuple[th.Tensor, dict[str, th.Tensor]]:
        with th.no_grad():
            self.log_temperature.data = th.maximum(self.log_temperature.data, self.min_log_temperature)
            self.log_alpha_mean.data = th.maximum(self.log_alpha_mean.data, self.min_log_alpha)
            self.log_alpha_stddev.data = th.maximum(self.log_alpha_stddev.data, self.min_log_alpha)
            if self._action_penalization:
                self._log_penalty_temperature.data = th.maximum(self._log_penalty_temperature.data, self.min_log_temperature)

        temperature = F.softplus(self.log_temperature) + _MPO_FLOAT_EPSILON
        alpha_mean = F.softplus(self.log_alpha_mean) + _MPO_FLOAT_EPSILON
        alpha_stddev = F.softplus(self.log_alpha_stddev) + _MPO_FLOAT_EPSILON

        weights, loss_temperature = _weights_and_temperature_loss(q_values, self._epsilon, temperature)
        kl_np = _nonparametric_kl_from_weights(weights)
        penalty_kl_np = None

        if self._action_penalization:
            pen_temp = F.softplus(self._log_penalty_temperature) + _MPO_FLOAT_EPSILON
            diff_oob = actions - actions.clamp(-1.0, 1.0)
            cost_oob = -diff_oob.norm(dim=-1)  # [N,B]
            pen_w, pen_temp_loss = _weights_and_temperature_loss(cost_oob, self._epsilon_penalty, pen_temp)
            penalty_kl_np = _nonparametric_kl_from_weights(pen_w)
            weights = weights + pen_w
            loss_temperature = loss_temperature + pen_temp_loss

        fixed_stddev = torch_dist.Independent(torch_dist.Normal(loc=online_mean, scale=target_std), 1)
        fixed_mean = torch_dist.Independent(torch_dist.Normal(loc=target_mean, scale=online_std), 1)

        loss_pi_mean = _cross_entropy_loss(actions, weights, fixed_stddev)
        loss_pi_std = _cross_entropy_loss(actions, weights, fixed_mean)

        kl_mean = _diag_normal_kl(target_mean, target_std, online_mean, target_std, self._per_dim)
        kl_std = _diag_normal_kl(target_mean, target_std, target_mean, online_std, self._per_dim)

        loss_kl_mean, loss_alpha_mean = _parametric_kl_penalty_and_dual(kl_mean, alpha_mean, self._epsilon_mean)
        loss_kl_std, loss_alpha_std = _parametric_kl_penalty_and_dual(kl_std, alpha_stddev, self._epsilon_stddev)

        loss_policy = loss_pi_mean + loss_pi_std
        loss_kl_penalty = loss_kl_mean + loss_kl_std
        loss_dual = loss_alpha_mean + loss_alpha_std + loss_temperature
        loss = loss_policy + loss_kl_penalty + loss_dual

        stats: dict[str, th.Tensor] = {
            "total_loss": loss.detach(),
            "loss_policy": loss_policy.detach(),
            "loss_temperature": loss_temperature.detach(),
            "loss_alpha": (loss_alpha_mean + loss_alpha_std).detach(),
            "dual_temperature": temperature.mean().detach(),
            "dual_alpha_mean": alpha_mean.mean().detach(),
            "dual_alpha_stddev": alpha_stddev.mean().detach(),
            "kl_q_rel": (kl_np.mean() / float(self._epsilon)).detach(),
            "kl_mean_rel": (kl_mean.mean() / float(self._epsilon_mean)).detach(),
            "kl_stddev_rel": (kl_std.mean() / float(self._epsilon_stddev)).detach(),
        }
        if penalty_kl_np is not None:
            stats["penalty_kl_q_rel"] = (penalty_kl_np.mean() / float(self._epsilon_penalty)).detach()
        return loss, stats


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (512, 512, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim + action_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

        # near-zero init on final layer for stability
        with th.no_grad():
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                nn.init.uniform_(final.weight, a=-1e-4, b=1e-4)
                nn.init.zeros_(final.bias)

    def forward(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        x = th.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


class MPO(OnPolicyAlgorithm):
    """
    MPO-style algorithm (maximum a posteriori policy optimisation).

    This implementation uses:
      - a replay buffer populated from on-policy rollouts,
      - a learned Q critic + target critic,
      - a target policy snapshot for sampling and KL constraints,
      - dual variables (temperature, alpha_mean, alpha_stddev) to enforce constraints.

    Note: This implementation supports only Box (continuous) action/observation spaces.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 1,  # kept for API compatibility; not used by MPO updates
        gamma: float = 0.99,
        gae_lambda: float = 0.95,  # kept for API compatibility
        clip_range: Union[float, Schedule] = 0.2,  # kept for API compatibility
        clip_range_vf: Union[None, float, Schedule] = None,  # kept for API compatibility
        normalize_advantage: bool = True,  # kept for API compatibility
        ent_coef: float = 0.0,  # kept for API compatibility
        vf_coef: float = 0.5,  # kept for API compatibility
        max_grad_norm: float = 40.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,  # kept for API compatibility; not used
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        # --- MPO-specific knobs ---
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
        gradient_steps: int = 1000,
        n_action_samples: int = 20,
        target_policy_update_interval: int = 25,
        target_critic_update_interval: int = 100,
        lr_dual: float = 1e-2,
        per_dim_constraining: bool = True,
        action_penalization: bool = False,
        separate_optimizers: bool = True,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box,),
        )

        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = bool(normalize_advantage)
        self.target_kl = target_kl
        self.separate_optimizers = bool(separate_optimizers)

        # MPO fields
        self.buffer_size = int(buffer_size)
        self.learning_starts = int(learning_starts)
        self.gradient_steps = int(gradient_steps)
        self.n_action_samples = int(n_action_samples)
        self.target_policy_update_interval = int(target_policy_update_interval)
        self.target_critic_update_interval = int(target_critic_update_interval)
        self.lr_dual = float(lr_dual)
        self.per_dim_constraining = bool(per_dim_constraining)
        self.action_penalization = bool(action_penalization)

        self.replay_buffer: Optional[ReplayBuffer] = None
        self.critic: Optional[QNetwork] = None
        self.critic_target: Optional[QNetwork] = None
        self.target_policy: Optional[BasePolicy] = None
        self.mpo_loss: Optional[MPOLoss] = None

        self.actor_optimizer: Optional[th.optim.Optimizer] = None
        self.critic_optimizer: Optional[th.optim.Optimizer] = None
        self.dual_optimizer: Optional[th.optim.Optimizer] = None

        self._learn_steps = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

        assert isinstance(self.action_space, spaces.Box), "This MPO implementation currently supports Box actions only."
        assert isinstance(self.observation_space, spaces.Box), (
            "This MPO implementation currently supports Box observations only "
            "(use a custom critic/policy for Dict/CNN observations)."
        )

        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))

        # Replay buffer fed from rollout transitions (derived from rollout_buffer)
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
        )

        # Critic + target critic
        self.critic = QNetwork(obs_dim, act_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Target policy snapshot (used for action sampling + MPO KL constraints)
        self.target_policy = copy.deepcopy(self.policy).to(self.device)
        self.target_policy.set_training_mode(False)

        # MPO loss + dual optimizer
        self.mpo_loss = MPOLoss(
            action_dim=act_dim,
            per_dim_constraining=self.per_dim_constraining,
            action_penalization=self.action_penalization,
        ).to(self.device)

        # Optimizers: actor uses SB3 policy optimizer hyperparams
        initial_lr = self.lr_schedule(1)
        if self.separate_optimizers:
            # Actor params: features extractor + policy MLP + action head (+ log_std)
            actor_params: list[th.nn.Parameter] = []
            seen = set()

            def _add(params_iter):
                nonlocal actor_params, seen
                for p in params_iter:
                    if id(p) not in seen:
                        actor_params.append(p)
                        seen.add(id(p))

            if getattr(self.policy, "share_features_extractor", True):
                _add(self.policy.features_extractor.parameters())
            else:
                _add(self.policy.pi_features_extractor.parameters())

            _add(self.policy.mlp_extractor.policy_net.parameters())
            _add(self.policy.action_net.parameters())
            if hasattr(self.policy, "log_std") and isinstance(self.policy.log_std, th.nn.Parameter):
                _add([self.policy.log_std])

            self.actor_optimizer = self.policy.optimizer_class(actor_params, lr=initial_lr, **self.policy.optimizer_kwargs)
        else:
            self.actor_optimizer = self.policy.optimizer_class(
                self.policy.parameters(), lr=initial_lr, **self.policy.optimizer_kwargs
            )

        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=initial_lr)
        self.dual_optimizer = th.optim.Adam(self.mpo_loss.parameters(), lr=self.lr_dual)

    def _rollout_to_replay(self) -> int:
        """
        Convert the last collected rollout into replay transitions.
        Uses:
          - obs_t: rollout_buffer.observations[t]
          - next_obs_t: rollout_buffer.observations[t+1] or self._last_obs for last step
          - done_t: rollout_buffer.episode_starts[t+1] or self._last_episode_starts for last step
        Returns number of transitions inserted (n_steps * n_envs).
        """
        assert self.replay_buffer is not None
        rb = self.rollout_buffer
        assert rb is not None
        assert self._last_obs is not None

        obs = rb.observations  # (n_steps, n_envs, *obs_shape)
        actions = rb.actions
        rewards = rb.rewards
        episode_starts = rb.episode_starts  # done flags for previous step

        n_steps, n_envs = rewards.shape[0], rewards.shape[1]

        # Build next_obs by shifting + last_obs
        next_obs = np.concatenate([obs[1:], self._last_obs[None, ...]], axis=0)

        # done_t := episode_starts[t+1] (or last_episode_starts for final step)
        last_dones = self._last_episode_starts.astype(np.float32)  # shape (n_envs,)
        dones = np.concatenate([episode_starts[1:].astype(np.float32), last_dones[None, :]], axis=0)

        # Insert per-timestep batches (vectorized env batch each add)
        empty_infos = [{} for _ in range(n_envs)]
        for t in range(n_steps):
            self.replay_buffer.add(
                obs[t],
                next_obs[t],
                actions[t],
                rewards[t],
                dones[t],
                empty_infos,
            )
        return int(n_steps * n_envs)

    def _actor_mean_std(self, obs_tensor: th.Tensor, *, use_target: bool) -> tuple[th.Tensor, th.Tensor]:
        pol = self.target_policy if use_target else self.policy
        assert pol is not None
        # Replay/env may provide float64; ensure it matches policy param dtype
        pol_dtype = next(pol.parameters()).dtype
        obs_tensor = obs_tensor.to(dtype=pol_dtype)
        d = pol.get_distribution(obs_tensor)
        indep = _as_independent_normal(d)
        mean = indep.base_dist.loc
        std = indep.base_dist.scale
        return mean, std

    def train(self) -> None:
        """
        MPO-style update using replay + target networks.

        High-level loop per gradient step:
          1) Critic: TD regression to a target computed from target critic and actions sampled from target policy.
          2) Actor/Duals: MPO loss (E-step weights from Q; M-step KL-constrained policy update via duals).
          3) Periodically hard-update target policy and target critic.
        """
        assert self.replay_buffer is not None
        assert self.critic is not None and self.critic_target is not None
        assert self.actor_optimizer is not None and self.critic_optimizer is not None and self.dual_optimizer is not None
        assert self.mpo_loss is not None and self.target_policy is not None

        # Switch policy to train mode for actor updates
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor_optimizer, self.critic_optimizer])

        # Feed latest rollout into replay
        inserted = self._rollout_to_replay()
        self.logger.record("train/replay_inserted", inserted, exclude="tensorboard")
        self.logger.record("train/replay_size", float(self.replay_buffer.size()))

        # Wait for enough data
        if self.replay_buffer.size() < max(self.learning_starts, self.batch_size):
            return

        # --- gradient updates ---
        critic_losses: list[float] = []
        policy_losses: list[float] = []
        q_mins: list[float] = []
        q_maxs: list[float] = []

        for _ in range(self.gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size)

            # Replay/env may provide float64; ensure it matches critic param dtype
            critic_dtype = next(self.critic.parameters()).dtype

            obs = batch.observations.to(dtype=critic_dtype)
            next_obs = batch.next_observations.to(dtype=critic_dtype)
            actions = batch.actions.to(dtype=critic_dtype)
            rewards = batch.rewards.to(dtype=critic_dtype).squeeze(-1)
            dones = batch.dones.to(dtype=critic_dtype).squeeze(-1)

            # Flatten obs for simple MLP critic (Box obs only)
            obs_f = obs.view(obs.shape[0], -1)
            next_obs_f = next_obs.view(next_obs.shape[0], -1)
            actions_f = actions.view(actions.shape[0], -1)

            # --- target Q (sample N actions from target policy) ---
            with th.no_grad():
                t_mean, t_std = self._actor_mean_std(next_obs, use_target=True)  # [B,D]
                N = self.n_action_samples
                eps = th.randn((N, t_mean.shape[0], t_mean.shape[1]), device=self.device, dtype=t_mean.dtype)
                sampled_actions = t_mean.unsqueeze(0) + t_std.unsqueeze(0) * eps  # [N,B,D]

                # clamp to action bounds
                low = th.as_tensor(self.action_space.low, device=self.device, dtype=sampled_actions.dtype)
                high = th.as_tensor(self.action_space.high, device=self.device, dtype=sampled_actions.dtype)
                sampled_actions = th.max(th.min(sampled_actions, high), low)

                next_obs_rep = next_obs_f.unsqueeze(0).expand(N, -1, -1).reshape(N * self.batch_size, -1)
                act_rep = sampled_actions.reshape(N * self.batch_size, -1)
                q_samples = self.critic_target(next_obs_rep, act_rep).view(N, self.batch_size)  # [N,B]
                q_samples = th.nan_to_num(q_samples, nan=0.0, posinf=1e6, neginf=-1e6)

                q_t = q_samples.mean(dim=0)  # [B]
                target_q = rewards + (1.0 - dones) * self.gamma * q_t

                q_mins.append(float(q_samples.min(dim=0).values.mean().item()))
                q_maxs.append(float(q_samples.max(dim=0).values.mean().item()))

            # --- critic update ---
            q_tm1 = self.critic(obs_f, actions_f)
            td = target_q - q_tm1
            critic_loss = 0.5 * (td.pow(2)).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            critic_losses.append(float(critic_loss.item()))

            # --- policy + dual update (MPO loss) ---
            with th.no_grad():
                # E-step uses Q(s,a) under target critic; sample actions from target policy at *current* obs
                t_mean_o, t_std_o = self._actor_mean_std(obs, use_target=True)
                eps_o = th.randn((N, t_mean_o.shape[0], t_mean_o.shape[1]), device=self.device, dtype=t_mean_o.dtype)
                sampled_actions_o = t_mean_o.unsqueeze(0) + t_std_o.unsqueeze(0) * eps_o
                sampled_actions_o = th.max(th.min(sampled_actions_o, high), low)

                obs_rep = obs_f.unsqueeze(0).expand(N, -1, -1).reshape(N * self.batch_size, -1)
                act_rep_o = sampled_actions_o.reshape(N * self.batch_size, -1)
                q_vals = self.critic_target(obs_rep, act_rep_o).view(N, self.batch_size)
                q_vals = th.nan_to_num(q_vals, nan=0.0, posinf=1e6, neginf=-1e6)

            online_mean, online_std = self._actor_mean_std(obs, use_target=False)

            loss_pi, stats = self.mpo_loss(
                actions=sampled_actions_o,
                q_values=q_vals,
                online_mean=online_mean,
                online_std=online_std,
                target_mean=t_mean_o,
                target_std=t_std_o,
            )

            self.actor_optimizer.zero_grad()
            self.dual_optimizer.zero_grad()
            loss_pi.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.dual_optimizer.step()
            policy_losses.append(float(loss_pi.item()))

            # --- periodic hard target updates ---
            self._learn_steps += 1
            if self._learn_steps % self.target_policy_update_interval == 0:
                self.target_policy.load_state_dict(self.policy.state_dict())
                self.target_policy.set_training_mode(False)
            if self._learn_steps % self.target_critic_update_interval == 0:
                self.critic_target.load_state_dict(self.critic.state_dict())
                for p in self.critic_target.parameters():
                    p.requires_grad_(False)

            # log a few key MPO stats (last minibatch wins)
            for k, v in stats.items():
                self.logger.record(f"train/{k}", float(v.item()) if isinstance(v, th.Tensor) else v)

        # Aggregate logs
        if critic_losses:
            self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        if policy_losses:
            self.logger.record("train/policy_loss", float(np.mean(policy_losses)))
        if q_mins:
            self.logger.record("train/q_min", float(np.mean(q_mins)))
        if q_maxs:
            self.logger.record("train/q_max", float(np.mean(q_maxs)))

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        # Include extra MPO components for checkpointing
        to_save = ["policy"]
        # replay buffer is handled by SB3 separately; keep it out of module state dict.
        to_save += ["critic", "critic_target", "target_policy", "mpo_loss"]
        to_save += ["actor_optimizer", "critic_optimizer", "dual_optimizer"]
        return to_save, []

    def learn(
        self: SelfMPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfMPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
