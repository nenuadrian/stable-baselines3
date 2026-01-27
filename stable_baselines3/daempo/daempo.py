import warnings
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np  # type: ignore[import]
import torch as th  # type: ignore[import]
import torch.nn as nn
from gymnasium import spaces  # type: ignore[import]
from torch.nn import functional as F  # type: ignore[import]

import copy
import torch.distributions as torch_dist  # type: ignore[import]
import contextlib

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from stable_baselines3.common.callbacks import BaseCallback

SelfDAEMPO = TypeVar("SelfDAEMPO", bound="DAEMPO")


@dataclass
class TrajBatch:
    # flat indices into flattened [T*N] for the first n steps (exclude endpoint)
    flat_indices: th.Tensor  # [B, n]
    observations: Any  # Tensor [B, n+1, ...] or dict[str, Tensor]
    actions: th.Tensor  # [B, n, ...]
    rewards: th.Tensor  # [B, n]


def _inv_softplus_scalar(x: float, device: th.device) -> th.Tensor:
    x_t = th.tensor(float(x), device=device)
    return th.log(th.expm1(x_t))


def _sb3_base_distribution(d: Any) -> Any:
    # SB3 Distribution wrappers usually expose `.distribution` (torch Distribution or list thereof)
    return getattr(d, "distribution", d)


def _sb3_dist_to_probs(d: Any) -> th.Tensor:
    """
    Convert SB3 distribution wrapper -> action probs (Discrete only).
    Returns probs with shape [B, n_actions]
    """
    dist = _sb3_base_distribution(d)
    if isinstance(dist, (list, tuple)):
        raise TypeError("DAE advantage centering supports Discrete only (no multi-distribution).")
    if isinstance(dist, torch_dist.Categorical):
        return dist.probs
    if hasattr(dist, "probs") and dist.probs is not None:
        return dist.probs
    if hasattr(dist, "logits") and dist.logits is not None:
        return F.softmax(dist.logits, dim=-1)
    raise TypeError(f"Unsupported distribution type for probs(): {type(dist)}")


class _DAEAdvantageHeadMixin:
    """
    Adds DAE advantage head A^θ(s,a) for Discrete actions via μ-centering.

    Uses the actor latent h(s) from SB3's ActorCriticPolicy family.
    """

    advantage_net: nn.Module

    def _build(self, lr_schedule: Schedule) -> None:  # type: ignore[override]
        super()._build(lr_schedule)  # type: ignore[misc]
        if not isinstance(self.action_space, spaces.Discrete):
            raise NotImplementedError("DAE advantage head currently supports Discrete action spaces only.")

        # Prefer SB3-provided attribute when available; otherwise infer via a dummy forward.
        latent_dim_pi = getattr(self.mlp_extractor, "latent_dim_pi", None)  # type: ignore[attr-defined]
        if latent_dim_pi is None:
            dummy_obs, _ = self.obs_to_tensor(self.observation_space.sample())
            if isinstance(dummy_obs, dict):
                dummy_obs = {k: v.to(self.device) for k, v in dummy_obs.items()}
            else:
                dummy_obs = dummy_obs.to(self.device)
            with th.no_grad():
                latent_pi = self._latent_pi(dummy_obs)
            latent_dim_pi = int(latent_pi.shape[-1])

        self.advantage_net = nn.Linear(int(latent_dim_pi), int(self.action_space.n))

    def _latent_pi(self, obs: th.Tensor) -> th.Tensor:
        # Use the same internal forward path as SB3 policies (robust across versions).
        features = self.extract_features(obs)  # SB3 handles dict observations too
        latent_pi, _latent_vf = self.mlp_extractor(features)  # type: ignore[misc]
        return latent_pi

    def _raw_advantages(self, obs: th.Tensor) -> th.Tensor:
        return self.advantage_net(self._latent_pi(obs))

    def centered_advantages(self, obs: th.Tensor, mu_dist: Optional[Any] = None) -> th.Tensor:
        """
        a_ctr(s) = a_raw(s) - sum_a' μ(a'|s) a_raw(s)[a']
        Returns: [B, n_actions]
        """
        if mu_dist is None:
            mu_dist = self.get_distribution(obs)
        a_raw = self._raw_advantages(obs)
        mu_probs = _sb3_dist_to_probs(mu_dist).to(dtype=a_raw.dtype)
        baseline = (mu_probs * a_raw).sum(dim=-1, keepdim=True)
        return a_raw - baseline

    def evaluate_advantage(self, obs: th.Tensor, actions: th.Tensor, mu_dist: Optional[Any] = None) -> th.Tensor:
        """
        Returns: A^θ(s,a) for executed actions, shape [B]
        """
        if actions.dim() > 1:
            actions = actions.view(-1)
        a_ctr = self.centered_advantages(obs, mu_dist=mu_dist)
        return a_ctr.gather(1, actions.long().view(-1, 1)).squeeze(1)


class DAEMPOActorCriticPolicy(_DAEAdvantageHeadMixin, ActorCriticPolicy):
    pass


class DAEMPOActorCriticCnnPolicy(_DAEAdvantageHeadMixin, ActorCriticCnnPolicy):
    pass


class DAEMPOMultiInputActorCriticPolicy(_DAEAdvantageHeadMixin, MultiInputActorCriticPolicy):
    pass


def _reduce_kl_to_batch(kl: th.Tensor) -> th.Tensor:
    # Want shape [B]; sum any extra event dims conservatively
    while kl.dim() > 1:
        kl = kl.sum(dim=-1)
    return kl


def _analytic_kl_sb3(target_dist: Any, online_dist: Any) -> th.Tensor:
    """
    Compute per-state KL: D_KL(p || q) with best-effort support for SB3 dist wrappers.
    Returns: kl_per_state with shape [B]

    Note: caller decides which direction by passing (p, q) in that order.
    """
    p = _sb3_base_distribution(target_dist)
    q = _sb3_base_distribution(online_dist)

    if isinstance(p, (list, tuple)):
        if not isinstance(q, (list, tuple)) or len(p) != len(q):
            raise TypeError(f"Incompatible distributions for KL: {type(p)} vs {type(q)}")
        kls = []
        for pi, qi in zip(p, q):
            kls.append(_reduce_kl_to_batch(torch_dist.kl_divergence(pi, qi)))
        return th.stack(kls, dim=0).sum(dim=0)

    kl = torch_dist.kl_divergence(p, q)
    return _reduce_kl_to_batch(kl)


def _split_diag_gaussian_kl(target_dist: Any, online_dist: Any) -> Optional[tuple[th.Tensor, th.Tensor]]:
    """
    Best-effort split KL for diagonal Gaussians into mean and std components.

    Computes KL(p || q) where p=target_dist and q=online_dist (in that order).
    Returns per-state tensors (kl_mean, kl_std) with shape [B], or None if unsupported.
    """
    p = _sb3_base_distribution(target_dist)
    q = _sb3_base_distribution(online_dist)

    if isinstance(p, torch_dist.Independent):
        p = p.base_dist
    if isinstance(q, torch_dist.Independent):
        q = q.base_dist

    if not isinstance(p, torch_dist.Normal) or not isinstance(q, torch_dist.Normal):
        return None

    # For KL(p||q): mean term uses q-variance; std term uses log(sigma_q/sigma_p) and var_p/var_q
    mu_p, mu_q = p.loc, q.loc
    sigma_p, sigma_q = p.scale, q.scale
    sigma_p = sigma_p.clamp_min(1e-6)
    sigma_q = sigma_q.clamp_min(1e-6)
    var_p, var_q = sigma_p**2, sigma_q**2

    mean_term = (mu_p - mu_q) ** 2 / (2.0 * var_q)
    std_term = (th.log(sigma_q) - th.log(sigma_p)) + (var_p / (2.0 * var_q)) - 0.5

    while mean_term.dim() > 1:
        mean_term = mean_term.sum(dim=-1)
    while std_term.dim() > 1:
        std_term = std_term.sum(dim=-1)

    return mean_term, std_term


class DAEMPO(OnPolicyAlgorithm):
    """
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param separate_optimizers: If True, use two optimizers to update actor and critic separately
        (hyperparameters identical). Shared feature extractor (if any) is updated once using
        the combined gradients from both losses.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        # Use DAEMPO policies so the advantage head exists by default.
        "MlpPolicy": DAEMPOActorCriticPolicy,
        "CnnPolicy": DAEMPOActorCriticCnnPolicy,
        "MultiInputPolicy": DAEMPOMultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        advantage_multiplier: float = 1.0,
        normalize_advantage_mean: bool = True,
        normalize_advantage_std: bool = True,
        separate_optimizers: bool = True,
        # --- DAE specific (trajectory minibatches) ---
        dae_backup_horizon: int = 32,
        trajectory_batch_size: Optional[int] = None,
        dae_beta: float = 1.0,
        dae_use_vf_loss: bool = False,
        # --- V-MPO specific ---
        epsilon_eta: float = 0.1,
        epsilon_alpha: float = 0.1,
        epsilon_alpha_mean: float = 0.02,
        epsilon_alpha_std: float = 0.02,
        init_eta: float = 1.0,
        init_alpha: float = 1.0,
        init_alpha_mean: float = 1.0,
        init_alpha_std: float = 1.0,
        top_adv_fraction: float = 0.5,
        eta_n_steps: int = 1,
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
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.advantage_multiplier = advantage_multiplier
        self.normalize_advantage_mean = normalize_advantage_mean
        self.normalize_advantage_std = normalize_advantage_std
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.separate_optimizers = separate_optimizers

        # --- DAE hyperparameters (persist, used in train()) ---
        self.dae_backup_horizon = int(dae_backup_horizon)
        self.trajectory_batch_size = trajectory_batch_size
        self.dae_beta = float(dae_beta)
        self.dae_use_vf_loss = bool(dae_use_vf_loss)

        # --- V-MPO hyperparameters ---
        assert epsilon_eta > 0, "`epsilon_eta` must be > 0"
        assert epsilon_alpha > 0, "`epsilon_alpha` must be > 0"
        assert init_eta > 0, "`init_eta` must be > 0"
        assert init_alpha > 0, "`init_alpha` must be > 0"
        assert 0 < top_adv_fraction <= 1.0, "`top_adv_fraction` must be in (0, 1]"
        assert eta_n_steps >= 1, "`eta_n_steps` must be >= 1"
        self.top_adv_fraction = float(top_adv_fraction)
        self.eta_n_steps = int(eta_n_steps)

        self.epsilon_eta = float(epsilon_eta)
        self.epsilon_alpha = float(epsilon_alpha)
        self.epsilon_alpha_mean = float(epsilon_alpha_mean)
        self.epsilon_alpha_std = float(epsilon_alpha_std)
        self.init_eta = float(init_eta)
        self.init_alpha = float(init_alpha)
        self.init_alpha_mean = float(init_alpha_mean)
        self.init_alpha_std = float(init_alpha_std)

        # Split-optimizer related attributes
        self.actor_optimizer: Optional[th.optim.Optimizer] = None
        self.critic_optimizer: Optional[th.optim.Optimizer] = None
        self.alpha_optimizer: Optional[th.optim.Optimizer] = None
        self.eta_optimizer: Optional[th.optim.Optimizer] = None  # dual optimizer for eta
        self._actor_params: Optional[list[th.nn.Parameter]] = None
        self._critic_params: Optional[list[th.nn.Parameter]] = None

        # Frozen snapshot used for the analytic distribution-KL constraint within each update
        self.target_policy: Optional[BasePolicy] = None

        self._last_dones: Optional[np.ndarray] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping (kept for backward compat; V-MPO does not use clip_range)
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

        # Register dual variables on the policy module (so they are checkpointed in policy.state_dict()).
        def _inv_softplus(x: float, device: th.device) -> th.Tensor:
            x_t = th.tensor(float(x), device=device)
            return th.log(th.expm1(x_t))

        if not hasattr(self.policy, "daempo_log_eta"):
            self.policy.daempo_log_eta = th.nn.Parameter(_inv_softplus(self.init_eta, self.device))  # type: ignore[attr-defined]
        if not hasattr(self.policy, "daempo_log_alpha"):
            self.policy.daempo_log_alpha = th.nn.Parameter(_inv_softplus(self.init_alpha, self.device))  # type: ignore[attr-defined]
        if not hasattr(self.policy, "daempo_log_alpha_mean"):
            self.policy.daempo_log_alpha_mean = th.nn.Parameter(_inv_softplus(self.init_alpha_mean, self.device))  # type: ignore[attr-defined]
        if not hasattr(self.policy, "daempo_log_alpha_std"):
            self.policy.daempo_log_alpha_std = th.nn.Parameter(_inv_softplus(self.init_alpha_std, self.device))  # type: ignore[attr-defined]

        # ensure dual params are never part of the base policy optimizer (non-split path)
        self._remove_dual_params_from_policy_optimizer()

        # When requested, build two optimizers with separated parameter groups
        if self.separate_optimizers:
            # Helpers to collect unique parameters
            def _extend_unique(dst: list[th.nn.Parameter], params_iter) -> None:
                seen = {id(p) for p in dst}
                for p in params_iter:
                    if id(p) not in seen:
                        dst.append(p)
                        seen.add(id(p))

            actor_params: list[th.nn.Parameter] = []
            critic_params: list[th.nn.Parameter] = []

            # Actor-specific modules
            _extend_unique(actor_params, self.policy.mlp_extractor.policy_net.parameters())
            _extend_unique(actor_params, self.policy.action_net.parameters())
            # NOTE: advantage_net belongs to DAE/critic updates (L_A), not the actor optimizer.
            # if hasattr(self.policy, "advantage_net"):
            #     _extend_unique(actor_params, self.policy.advantage_net.parameters())
            if hasattr(self.policy, "log_std") and isinstance(self.policy.log_std, th.nn.Parameter):
                actor_params.append(self.policy.log_std)

            # Critic-specific modules
            _extend_unique(critic_params, self.policy.mlp_extractor.value_net.parameters())
            _extend_unique(critic_params, self.policy.value_net.parameters())
            if hasattr(self.policy, "advantage_net"):
                _extend_unique(critic_params, self.policy.advantage_net.parameters())  # DAE updates

            # Feature extractor: update it ONCE with combined gradients (actor+critic).
            if getattr(self.policy, "share_features_extractor", True):
                _extend_unique(critic_params, self.policy.features_extractor.parameters())
            else:
                _extend_unique(actor_params, self.policy.pi_features_extractor.parameters())
                _extend_unique(critic_params, self.policy.vf_features_extractor.parameters())

            # Dual variables: do NOT optimize η with the actor loss; update it via its own dual objective.
            # (Keep α updated separately too.)
            self._actor_params = actor_params
            self._critic_params = critic_params

            initial_lr = self.lr_schedule(1)
            alpha_lr = initial_lr * 0.1
            eta_lr = initial_lr * 0.1

            self.actor_optimizer = self.policy.optimizer_class(self._actor_params, lr=initial_lr, **self.policy.optimizer_kwargs)  # type: ignore[arg-type]
            self.critic_optimizer = self.policy.optimizer_class(self._critic_params, lr=initial_lr, **self.policy.optimizer_kwargs)  # type: ignore[arg-type]
            self.alpha_optimizer = self.policy.optimizer_class(
                [self.policy.daempo_log_alpha, self.policy.daempo_log_alpha_mean, self.policy.daempo_log_alpha_std],
                lr=alpha_lr,
                **self.policy.optimizer_kwargs,
            )  # type: ignore[arg-type,attr-defined]
            self.eta_optimizer = self.policy.optimizer_class(
                [self.policy.daempo_log_eta],
                lr=eta_lr,
                **self.policy.optimizer_kwargs,
            )  # type: ignore[arg-type,attr-defined]
        else:
            # Ensure dual params are NOT optimized via the single policy optimizer.
            initial_lr = self.lr_schedule(1)
            alpha_lr = initial_lr * 0.1
            eta_lr = initial_lr * 0.1
            self.alpha_optimizer = self.policy.optimizer_class(
                [self.policy.daempo_log_alpha, self.policy.daempo_log_alpha_mean, self.policy.daempo_log_alpha_std],
                lr=alpha_lr,
                **self.policy.optimizer_kwargs,
            )  # type: ignore[arg-type,attr-defined]
            self.eta_optimizer = self.policy.optimizer_class(
                [self.policy.daempo_log_eta],
                lr=eta_lr,
                **self.policy.optimizer_kwargs,
            )  # type: ignore[arg-type,attr-defined]

        # Create target policy once; synced at the start of each train() call (per update)
        if self.target_policy is None:
            self.target_policy = copy.deepcopy(self.policy).to(self.device)
        if self.target_policy is not None:
            self.target_policy.set_training_mode(False)

    def _iter_trajectory_minibatches(self, n: int, b_traj: int):
        """
        Yield valid trajectory segments of length n transitions with an endpoint observation at t+n.

        - rb.* are indexed as [T, N, ...]
        - observations returned as [B, n+1, ...] (or dict of that)
        - endpoint at t+n comes from buffer if t+n < T else from self._last_obs
        - segments never cross episode boundaries (uses episode_starts + self._last_dones)
        """
        rb = self.rollout_buffer
        T = int(rb.rewards.shape[0])
        N = int(rb.rewards.shape[1])

        if n <= 0 or b_traj <= 0 or n > T:
            return

        starts = th.as_tensor(rb.episode_starts, device=self.device).bool()  # [T, N]
        done = th.zeros((T, N), device=self.device, dtype=th.bool)
        if T > 1:
            done[:-1] = starts[1:]  # transition at t ends episode if next step is episode start
        done[-1] = (
            th.as_tensor(self._last_dones, device=self.device).bool()
            if self._last_dones is not None
            else th.zeros((N,), device=self.device, dtype=th.bool)
        )

        # valid start positions (t0, e0) s.t. no done in [t0, t0+n-1]
        window = done.unfold(dimension=0, size=n, step=1)  # [T-n+1, N, n]
        valid = ~window.any(dim=-1)  # [T-n+1, N]
        idx = valid.nonzero(as_tuple=False)  # [M, 2] columns: (t0, e0)
        if idx.numel() == 0:
            return

        idx = idx[th.randperm(idx.shape[0], device=self.device)]

        def _as_t(x):
            return th.as_tensor(x, device=self.device)

        obs_buf = rb.observations
        act_buf = _as_t(rb.actions)
        rew_buf = _as_t(rb.rewards)

        last_obs = self._last_obs
        if isinstance(last_obs, dict):
            last_obs_t = {k: _as_t(v) for k, v in last_obs.items()}  # each [N, ...]
        else:
            last_obs_t = _as_t(last_obs)  # [N, ...]

        arange_n = th.arange(n, device=self.device, dtype=th.long)

        for i in range(0, idx.shape[0], b_traj):
            batch = idx[i : i + b_traj]
            t0 = batch[:, 0].long()  # [B]
            e0 = batch[:, 1].long()  # [B]
            B = int(t0.shape[0])

            tt = t0[:, None] + arange_n[None, :]  # [B, n]
            t_end = t0 + n  # [B] endpoint index (may be == T)
            mask_in_buf = t_end < T  # [B]

            flat = (tt * N + e0[:, None]).long()  # [B, n]

            # observations: [B, n, ...] + endpoint [B, ...] -> [B, n+1, ...]
            if isinstance(obs_buf, dict):
                obs_seq: dict[str, th.Tensor] = {}
                for k, v in obs_buf.items():
                    v_t = _as_t(v)  # [T, N, ...]
                    obs_steps = v_t[tt, e0[:, None]]  # [B, n, ...]
                    end = th.empty((B, *v_t.shape[2:]), device=self.device, dtype=v_t.dtype)
                    if mask_in_buf.any():
                        end[mask_in_buf] = v_t[t_end[mask_in_buf], e0[mask_in_buf]]
                    if (~mask_in_buf).any():
                        end[~mask_in_buf] = last_obs_t[k][e0[~mask_in_buf]]
                    obs_seq[k] = th.cat([obs_steps, end[:, None]], dim=1)  # [B, n+1, ...]
            else:
                v_t = _as_t(obs_buf)  # [T, N, ...]
                obs_steps = v_t[tt, e0[:, None]]  # [B, n, ...]
                end = th.empty((B, *v_t.shape[2:]), device=self.device, dtype=v_t.dtype)
                if mask_in_buf.any():
                    end[mask_in_buf] = v_t[t_end[mask_in_buf], e0[mask_in_buf]]
                if (~mask_in_buf).any():
                    end[~mask_in_buf] = last_obs_t[e0[~mask_in_buf]]
                obs_seq = th.cat([obs_steps, end[:, None]], dim=1)  # [B, n+1, ...]

            actions = act_buf[tt, e0[:, None]]  # [B, n, ...]
            rewards = rew_buf[tt, e0[:, None]]  # [B, n] or [B, n, 1] depending on buffer

            yield TrajBatch(flat_indices=flat, observations=obs_seq, actions=actions, rewards=rewards)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer. (V-MPO update)
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Sync (freeze) target policy for this whole update: π_tgt := π_old
        assert self.target_policy is not None
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.set_training_mode(False)

        entropy_losses = []
        pg_losses, value_losses = [], []
        dae_losses = []
        approx_kl_divs = []
        eta_losses, alpha_losses = [], []
        alpha_dual_mean, alpha_dual_std = [], []
        kls = []
        kl_means, kl_stds = [], []
        kl_split_available = []
        weight_entropies = []
        weight_ess = []
        grad_norms = []
        actor_grad_norms: list[float] = []
        critic_grad_norms: list[float] = []
        batch_advantages = []
        batch_norm_advantages = []
        continue_training = True

        if not hasattr(self.policy, "evaluate_advantage"):
            raise RuntimeError("DAE requires a DAEMPO*Policy with `evaluate_advantage()` (Discrete only).")

        # --- Compute A_hat (detached) for the whole rollout, using μ := target_policy ---
        rb = self.rollout_buffer

        def _flatten_obs(obs_buf):
            if isinstance(obs_buf, dict):
                return {k: th.as_tensor(v.reshape(-1, *v.shape[2:]), device=self.device) for k, v in obs_buf.items()}
            return th.as_tensor(obs_buf.reshape(-1, *obs_buf.shape[2:]), device=self.device)

        def _index_select_obs(obs_flat: Any, idx_1d: th.Tensor) -> Any:
            if isinstance(obs_flat, dict):
                return {k: v.index_select(0, idx_1d) for k, v in obs_flat.items()}
            return obs_flat.index_select(0, idx_1d)

        obs_all = _flatten_obs(rb.observations)
        actions_all = th.as_tensor(rb.actions.reshape(-1, *rb.actions.shape[2:]), device=self.device)
        if isinstance(self.action_space, spaces.Discrete):
            actions_all = actions_all.long().view(-1)

        with th.no_grad():
            mu_dist_all = self.target_policy.get_distribution(obs_all)
        adv_detached_all = self.target_policy.evaluate_advantage(obs_all, actions_all, mu_dist=mu_dist_all).detach()  # type: ignore[attr-defined]

        # --- E-step "advantages" := A_hat_detached (optionally normalized) ---
        advantages_estep_all = adv_detached_all

        adv_mean_t: Optional[th.Tensor] = None
        adv_std_t: Optional[th.Tensor] = None
        if self.normalize_advantage and advantages_estep_all.numel() > 1:
            if self.normalize_advantage_mean:
                adv_mean_t = advantages_estep_all.mean()
                advantages_estep_all = advantages_estep_all - adv_mean_t
            if self.normalize_advantage_std:
                adv_std_t = advantages_estep_all.std(unbiased=False) + 1e-8
                advantages_estep_all = advantages_estep_all / adv_std_t
            batch_norm_advantages.append(advantages_estep_all.detach().cpu().numpy())

        advantages_estep_all = advantages_estep_all * self.advantage_multiplier
        batch_advantages.append(advantages_estep_all.detach().cpu().numpy())

        total_samples = int(advantages_estep_all.shape[0])
        k = max(1, int(self.top_adv_fraction * total_samples))

        sel_adv, _ = th.topk(advantages_estep_all, k)
        adv_threshold = sel_adv.min()
        mask_all = advantages_estep_all >= adv_threshold
        selected_adv = advantages_estep_all[mask_all]
        k_eff = selected_adv.numel()
        selected_adv_detached = selected_adv.detach()

        with th.no_grad():
            denom = th.log(th.tensor(float(k_eff), device=selected_adv.device, dtype=selected_adv.dtype))

        # --- Dual ascent for eta (global objective) using detached A_hat ---
        if self.eta_optimizer is not None:
            for _ in range(self.eta_n_steps):
                eta = (F.softplus(self.policy.daempo_log_eta) + 1e-8).clamp_min(1e-3)  # type: ignore[attr-defined]
                logZ_mean = th.logsumexp(selected_adv_detached / eta, dim=0) - denom
                eta_loss_global = eta * (self.epsilon_eta + logZ_mean)

                self._update_learning_rate(self.eta_optimizer)
                self.eta_optimizer.zero_grad()
                eta_loss_global.backward()
                self.eta_optimizer.step()
                eta_losses.append(float(eta_loss_global.detach().cpu().item()))

        eta_fixed = (F.softplus(self.policy.daempo_log_eta) + 1e-8).clamp_min(1e-3).detach()  # type: ignore[attr-defined]
        logZ_fixed = th.logsumexp(selected_adv_detached / eta_fixed, dim=0) - denom

        # Clamp log-weights before exponentiating to avoid inf/NaN
        log_w_all = advantages_estep_all / eta_fixed - logZ_fixed
        log_w_all = log_w_all.clamp(min=-50.0, max=50.0)
        weights_all_detached = th.exp(log_w_all) * mask_all
        weights_all_detached = weights_all_detached.detach()

        def _compute_dae_loss(obs_mb_flat, actions_mb_flat, rewards_seq, obs_end, B: int, n: int) -> th.Tensor:
            with th.no_grad():
                mu_dist_steps = self.target_policy.get_distribution(obs_mb_flat)
                v_target_end = self.target_policy.predict_values(obs_end).flatten().float()
            a_hat_flat = self.policy.evaluate_advantage(obs_mb_flat, actions_mb_flat, mu_dist=mu_dist_steps)  # type: ignore[attr-defined]
            a_hat = a_hat_flat.view(B, n).float()
            v_hat = self.policy.predict_values(obs_mb_flat).flatten().view(B, n).float()

            r_tilde = rewards_seq - a_hat  # [B, n] (FP32)
            Z = th.empty((B, n), device=self.device, dtype=th.float32)
            z_next = v_target_end  # [B] (FP32)
            for t in range(n - 1, -1, -1):
                z_next = r_tilde[:, t] + self.gamma * z_next
                Z[:, t] = z_next

            return (Z - v_hat).pow(2).mean()

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            b_traj = self.trajectory_batch_size
            if b_traj is None:
                b_traj = max(1, int(self.batch_size // max(1, self.dae_backup_horizon)))

            for traj_data in self._iter_trajectory_minibatches(self.dae_backup_horizon, b_traj):
                B = int(traj_data.flat_indices.shape[0])
                n = int(traj_data.flat_indices.shape[1])

                flat_idx = traj_data.flat_indices.reshape(-1)  # [B*n]
                # --- NEW: gather precomputed weights per (state, action) sample ---
                weights_flat = weights_all_detached.index_select(0, flat_idx)

                # Endpoint obs stays trajectory-shaped (needed for recursion bootstrap)
                if isinstance(traj_data.observations, dict):
                    obs_end = {k: v[:, -1] for k, v in traj_data.observations.items()}  # [B, ...]
                else:
                    obs_end = traj_data.observations[:, -1]  # [B, ...]

                # ===== Evaluate policy on flattened rollout samples selected by flat_idx =====
                obs_mb_flat = _index_select_obs(obs_all, flat_idx)  # [B*n, ...]
                actions_mb_flat = actions_all.index_select(0, flat_idx)  # [B*n, ...]
                values, log_prob, entropy = self.policy.evaluate_actions(obs_mb_flat, actions_mb_flat)
                values = values.flatten()
                log_prob = log_prob.flatten()

                weights_flat = weights_flat.to(device=log_prob.device, dtype=log_prob.dtype).reshape(-1)
                log_prob_flat = log_prob.reshape(-1)
                if weights_flat.shape[0] != log_prob_flat.shape[0]:
                    raise RuntimeError(
                        f"DAEMPO shape mismatch: weights_flat={tuple(weights_flat.shape)} "
                        f"log_prob={tuple(log_prob_flat.shape)}; expected both to be [B*n]."
                    )

                w = weights_flat
                w_sum = w.sum().clamp_min(1e-8)
                w_norm = w / w_sum
                policy_loss = -(w_norm * log_prob_flat).sum()
                pg_losses.append(policy_loss.item())

                with th.no_grad():
                    tgt_dist = self.target_policy.get_distribution(obs_mb_flat)  # old (μ)
                online_dist = self.policy.get_distribution(obs_mb_flat)
                kl_vec = _analytic_kl_sb3(online_dist, tgt_dist)
                kl = kl_vec.mean()
                kls.append(kl.detach().cpu().item())
                approx_kl_divs.append(kl.detach().cpu().item())

                if self.target_kl is not None and kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                split_kl = _split_diag_gaussian_kl(online_dist, tgt_dist)
                kl_split_available.append(float(split_kl is not None))
                if split_kl is not None:
                    kl_mean_vec, kl_std_vec = split_kl
                    kl_mean = kl_mean_vec.mean()
                    kl_std = kl_std_vec.mean()
                    kl_means.append(kl_mean.detach().cpu().item())
                    kl_stds.append(kl_std.detach().cpu().item())

                    alpha_m = F.softplus(self.policy.daempo_log_alpha_mean) + 1e-8  # type: ignore[attr-defined]
                    alpha_s = F.softplus(self.policy.daempo_log_alpha_std) + 1e-8  # type: ignore[attr-defined]
                    alpha_m_dual = alpha_m * (kl_mean.detach() - self.epsilon_alpha_mean)
                    alpha_s_dual = alpha_s * (kl_std.detach() - self.epsilon_alpha_std)
                    alpha_dual_mean.append(alpha_m_dual.item())
                    alpha_dual_std.append(alpha_s_dual.item())
                    alpha_losses.append((alpha_m_dual + alpha_s_dual).item())
                    kl_penalty = alpha_m.detach() * kl_mean + alpha_s.detach() * kl_std
                else:
                    alpha = F.softplus(self.policy.daempo_log_alpha) + 1e-8  # type: ignore[attr-defined]
                    alpha_dual = alpha * (kl.detach() - self.epsilon_alpha)
                    alpha_losses.append(alpha_dual.item())
                    kl_penalty = alpha.detach() * kl

                # ===== DAE loss L_A (non-detached A_hat inside L_A) =====
                rewards_seq = traj_data.rewards.squeeze(-1) if traj_data.rewards.dim() == 3 else traj_data.rewards  # [B, n]
                rewards_seq = rewards_seq.float()  # keep recursion stable (avoid FP16 under AMP)

                # ===== Total loss components (actor side) =====
                entropy_loss = -th.mean(entropy) if entropy is not None else th.zeros((), device=self.device)
                entropy_losses.append(float(entropy_loss.detach().cpu().item()))

                actor_total_loss = policy_loss + self.ent_coef * entropy_loss + kl_penalty

                if self.separate_optimizers:
                    # --- ACTOR STEP ---
                    assert self.actor_optimizer is not None
                    assert self._actor_params is not None
                    assert self._critic_params is not None

                    self._update_learning_rate(self.actor_optimizer)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    with self._temporarily_disable_grad(self._critic_params):
                        actor_total_loss.backward()
                    actor_gn = self._clip_and_log_grad_norm(self._actor_params, self.max_grad_norm)
                    actor_grad_norms.append(actor_gn)
                    self.actor_optimizer.step()

                    # --- CRITIC (VALUE + ADV HEAD) STEP ---
                    assert self.critic_optimizer is not None
                    dae_loss = _compute_dae_loss(obs_mb_flat, actions_mb_flat, rewards_seq, obs_end, B, n)
                    dae_losses.append(float(dae_loss.detach().cpu().item()))
                    critic_total_loss = self.dae_beta * dae_loss
                    if self.dae_use_vf_loss:
                        value_losses.append(float("nan"))

                    self._update_learning_rate(self.critic_optimizer)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    with self._temporarily_disable_grad(self._actor_params):
                        critic_total_loss.backward()
                    critic_gn = self._clip_and_log_grad_norm(self._critic_params, self.max_grad_norm)
                    critic_grad_norms.append(critic_gn)
                    self.critic_optimizer.step()
                else:
                    # ===== DAE loss L_A (single optimizer path) =====
                    dae_loss = _compute_dae_loss(obs_mb_flat, actions_mb_flat, rewards_seq, obs_end, B, n)
                    dae_losses.append(float(dae_loss.detach().cpu().item()))

                    critic_total_loss = self.dae_beta * dae_loss
                    if self.dae_use_vf_loss:
                        # keep old value MSE term (generally discouraged for "pure" DAE)
                        # ...existing code for values_pred/returns_flat if you want it; keep minimal here...
                        value_losses.append(float("nan"))
                    total_loss = actor_total_loss + critic_total_loss

                    # --- SINGLE POLICY OPTIMIZER STEP ---
                    assert hasattr(self.policy, "optimizer") and self.policy.optimizer is not None
                    self._remove_dual_params_from_policy_optimizer()  # safety (in case something re-added them)

                    self._update_learning_rate(self.policy.optimizer)
                    self.policy.optimizer.zero_grad(set_to_none=True)

                    total_loss.backward()

                    base_params = self._optimizer_params(self.policy.optimizer)
                    gn = self._clip_and_log_grad_norm(base_params, self.max_grad_norm)
                    grad_norms.append(gn)

                    self.policy.optimizer.step()

                # Dual ascent step for alpha (optimizer minimizes, so minimize NEGATIVE dual objective)
                if self.alpha_optimizer is not None:
                    self._update_learning_rate(self.alpha_optimizer)
                    self.alpha_optimizer.zero_grad(set_to_none=True)

                    if split_kl is not None:
                        alpha_m = F.softplus(self.policy.daempo_log_alpha_mean) + 1e-8  # type: ignore[attr-defined]
                        alpha_s = F.softplus(self.policy.daempo_log_alpha_std) + 1e-8  # type: ignore[attr-defined]
                        alpha_opt_loss = -(
                            alpha_m * (kl_mean.detach() - self.epsilon_alpha_mean)
                            + alpha_s * (kl_std.detach() - self.epsilon_alpha_std)
                        )
                    else:
                        alpha = F.softplus(self.policy.daempo_log_alpha) + 1e-8  # type: ignore[attr-defined]
                        alpha_opt_loss = -(alpha * (kl.detach() - self.epsilon_alpha))

                    alpha_opt_loss.backward()
                    self.alpha_optimizer.step()

                    # Clamp dual variables to keep KL penalty effective
                    alpha_min = 1e-3
                    with th.no_grad():
                        clamp_min = _inv_softplus_scalar(alpha_min, self.device)
                        if hasattr(self.policy, "daempo_log_alpha"):
                            self.policy.daempo_log_alpha.data.clamp_(min=clamp_min)  # type: ignore[attr-defined]
                        if hasattr(self.policy, "daempo_log_alpha_mean"):
                            self.policy.daempo_log_alpha_mean.data.clamp_(min=clamp_min)  # type: ignore[attr-defined]
                        if hasattr(self.policy, "daempo_log_alpha_std"):
                            self.policy.daempo_log_alpha_std.data.clamp_(min=clamp_min)  # type: ignore[attr-defined]

            # --- END epoch ---
            self._n_updates += 1
            if not continue_training:
                break

        # --- End of training loop ---
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        if len(batch_advantages) > 0:
            adv = np.concatenate([a.reshape(-1) for a in batch_advantages])
            self.logger.record("train/advantages_mean", float(adv.mean()))
            self.logger.record("train/advantages_sum", float(adv.sum()))
        if len(batch_norm_advantages) > 0:
            advn = np.concatenate([a.reshape(-1) for a in batch_norm_advantages])
            self.logger.record("train/advantages_norm_mean", float(advn.mean()))
            self.logger.record("train/advantages_norm_sum", float(advn.sum()))

        def _log_mean(name: str, xs) -> None:
            if len(xs) > 0:
                self.logger.record(name, float(np.mean(xs)))

        def _log_max(name: str, xs) -> None:
            if len(xs) > 0:
                self.logger.record(name, float(np.max(xs)))

        _log_mean("train/policy_loss_daempo", pg_losses)
        _log_mean("train/value_loss", value_losses)
        _log_mean("train/entropy_loss", entropy_losses)
        _log_mean("train/dae_loss", dae_losses)

        _log_mean("train/kl_pi_new_pi_old", kls)

        self.logger.record(
            "train/kl_split_available", float(np.mean(kl_split_available)) if len(kl_split_available) > 0 else 0.0
        )
        _log_mean("train/kl_mean", kl_means)
        _log_mean("train/kl_std", kl_stds)
        _log_mean("train/approx_kl", approx_kl_divs)
        _log_mean("train/eta_loss", eta_losses)
        _log_mean("train/alpha_dual", alpha_losses)
        _log_mean("train/alpha_dual_mean", alpha_dual_mean)
        _log_mean("train/alpha_dual_std", alpha_dual_std)
        _log_mean("train/weights_entropy", weight_entropies)
        _log_mean("train/weights_eff_sample_size", weight_ess)

        # Dual variable values
        if hasattr(self.policy, "daempo_log_eta"):
            eta_val = (F.softplus(self.policy.daempo_log_eta) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
            self.logger.record("train/eta", eta_val)
        if hasattr(self.policy, "daempo_log_alpha"):
            alpha_val = (F.softplus(self.policy.daempo_log_alpha) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
            self.logger.record("train/alpha", alpha_val)
        if hasattr(self.policy, "daempo_log_alpha_mean"):
            alpha_mean_val = (F.softplus(self.policy.daempo_log_alpha_mean) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
            self.logger.record("train/alpha_mean", alpha_mean_val)
        if hasattr(self.policy, "daempo_log_alpha_std"):
            alpha_std_val = (F.softplus(self.policy.daempo_log_alpha_std) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
            self.logger.record("train/alpha_std", alpha_std_val)

        self.logger.record("train/explained_variance", explained_var)

        if len(actor_grad_norms) > 0:
            _log_mean("train/grad_norm_actor", actor_grad_norms)
            _log_max("train/grad_norm_actor/max", actor_grad_norms)
        if len(critic_grad_norms) > 0:
            _log_mean("train/grad_norm_critic", critic_grad_norms)
            _log_max("train/grad_norm_critic/max", critic_grad_norms)
        if len(grad_norms) > 0:
            _log_mean("train/grad_norm", grad_norms)
            _log_max("train/grad_norm/max", grad_norms)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def _clamp_log_std(self) -> None:
        if hasattr(self.policy, "log_std"):
            with th.no_grad():
                self.policy.log_std.data.clamp_(min=float(np.log(0.2)), max=float(np.log(2.0)))

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Include separate optimizers in state dicts when enabled.
        """
        if self.separate_optimizers:
            return [
                "policy",
                "policy.optimizer",
                "actor_optimizer",
                "critic_optimizer",
                "alpha_optimizer",
                "eta_optimizer",
            ], []
        # Default behavior from parent
        return super()._get_torch_save_params()

    def learn(
        self: SelfDAEMPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "DAEMPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDAEMPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    # --- helpers (must be methods; used via self.* above) ---

    @staticmethod
    def _unique_params(params) -> list[th.nn.Parameter]:
        uniq: list[th.nn.Parameter] = []
        seen: set[int] = set()
        for p in params:
            if p is None:
                continue
            pid = id(p)
            if pid not in seen:
                uniq.append(p)
                seen.add(pid)
        return uniq

    def _optimizer_params(self, optimizer: th.optim.Optimizer) -> list[th.nn.Parameter]:
        params: list[th.nn.Parameter] = []
        for g in optimizer.param_groups:
            params.extend(g.get("params", []))
        return self._unique_params(params)

    def _clip_and_log_grad_norm(self, params, max_norm: float) -> float:
        if max_norm is None or max_norm <= 0:
            return float("nan")
        params_u = [p for p in self._unique_params(params) if p.requires_grad]
        if len(params_u) == 0:
            return float("nan")
        gn = th.nn.utils.clip_grad_norm_(params_u, max_norm)
        return float(gn.detach().cpu().item())

    @contextlib.contextmanager
    def _temporarily_disable_grad(self, params):
        params_u = self._unique_params(params)
        old = [p.requires_grad for p in params_u]
        try:
            for p in params_u:
                p.requires_grad_(False)
            yield
        finally:
            for p, req in zip(params_u, old):
                p.requires_grad_(req)

    def _remove_dual_params_from_policy_optimizer(self) -> None:
        opt = getattr(self.policy, "optimizer", None)
        if opt is None:
            return

        duals: list[th.nn.Parameter] = []
        for name in ("daempo_log_eta", "daempo_log_alpha", "daempo_log_alpha_mean", "daempo_log_alpha_std"):
            if hasattr(self.policy, name):
                duals.append(getattr(self.policy, name))

        if len(duals) == 0:
            return

        dual_ids = {id(p) for p in duals}
        for g in opt.param_groups:
            g["params"] = [p for p in g.get("params", []) if id(p) not in dual_ids]
