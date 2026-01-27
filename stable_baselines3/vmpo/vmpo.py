import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np  # type: ignore[import]
import torch as th  # type: ignore[import]
from gymnasium import spaces  # type: ignore[import]
from torch.nn import functional as F  # type: ignore[import]

import copy
import torch.distributions as torch_dist  # type: ignore[import]

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance

SelfVMPO = TypeVar("SelfVMPO", bound="VMPO")


def _sb3_base_distribution(d: Any) -> Any:
    # SB3 Distribution wrappers usually expose `.distribution` (torch Distribution or list thereof)
    return getattr(d, "distribution", d)


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


class VMPO(OnPolicyAlgorithm):
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
        eta_n_steps: int = 1,  # NEW: explicit eta dual steps per iteration
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
        self.eta_optimizer: Optional[th.optim.Optimizer] = None  # NEW: dual optimizer for eta
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

        if not hasattr(self.policy, "vmpo_log_eta"):
            self.policy.vmpo_log_eta = th.nn.Parameter(_inv_softplus(self.init_eta, self.device))  # type: ignore[attr-defined]
        if not hasattr(self.policy, "vmpo_log_alpha"):
            self.policy.vmpo_log_alpha = th.nn.Parameter(_inv_softplus(self.init_alpha, self.device))  # type: ignore[attr-defined]
        if not hasattr(self.policy, "vmpo_log_alpha_mean"):
            self.policy.vmpo_log_alpha_mean = th.nn.Parameter(_inv_softplus(self.init_alpha_mean, self.device))  # type: ignore[attr-defined]
        if not hasattr(self.policy, "vmpo_log_alpha_std"):
            self.policy.vmpo_log_alpha_std = th.nn.Parameter(_inv_softplus(self.init_alpha_std, self.device))  # type: ignore[attr-defined]

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
            if hasattr(self.policy, "log_std") and isinstance(self.policy.log_std, th.nn.Parameter):
                actor_params.append(self.policy.log_std)

            # Critic-specific modules
            _extend_unique(critic_params, self.policy.mlp_extractor.value_net.parameters())
            _extend_unique(critic_params, self.policy.value_net.parameters())

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
                [self.policy.vmpo_log_alpha, self.policy.vmpo_log_alpha_mean, self.policy.vmpo_log_alpha_std],
                lr=alpha_lr,
                **self.policy.optimizer_kwargs,
            )  # type: ignore[arg-type,attr-defined]
            self.eta_optimizer = self.policy.optimizer_class(
                [self.policy.vmpo_log_eta],
                lr=eta_lr,
                **self.policy.optimizer_kwargs,
            )  # type: ignore[arg-type,attr-defined]
        else:
            # Ensure dual params are NOT optimized via the single policy optimizer.
            initial_lr = self.lr_schedule(1)
            alpha_lr = initial_lr * 0.1
            eta_lr = initial_lr * 0.1
            self.alpha_optimizer = self.policy.optimizer_class(
                [self.policy.vmpo_log_alpha, self.policy.vmpo_log_alpha_mean, self.policy.vmpo_log_alpha_std],
                lr=alpha_lr,
                **self.policy.optimizer_kwargs,
            )  # type: ignore[arg-type,attr-defined]
            self.eta_optimizer = self.policy.optimizer_class(
                [self.policy.vmpo_log_eta],
                lr=eta_lr,
                **self.policy.optimizer_kwargs,
            )  # type: ignore[arg-type,attr-defined]

        # Create target policy once; synced at the start of each train() call (per update)
        if self.target_policy is None:
            self.target_policy = copy.deepcopy(self.policy).to(self.device)
        if self.target_policy is not None:
            self.target_policy.set_training_mode(False)

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
        approx_kl_divs = []
        eta_losses, alpha_losses = [], []
        alpha_dual_mean, alpha_dual_std = [], []
        kls = []
        kl_means, kl_stds = [], []
        kl_split_available = []
        weight_entropies = []
        weight_ess = []
        grad_norms = []
        actor_grad_norms: list[np.ndarray] = []
        critic_grad_norms: list[np.ndarray] = []
        batch_advantages = []
        batch_norm_advantages = []
        continue_training = True

        # --- Precompute advantages/top-k/weights once per update ---
        advantages_all = th.as_tensor(self.rollout_buffer.advantages, device=self.device).flatten()
        advantages_estep_all = advantages_all

        # --- Normalize advantages ---
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

        # --- Dual ascent for eta (global objective) ONCE per iteration, with explicit inner steps ---
        if self.eta_optimizer is not None:
            for _ in range(self.eta_n_steps):
                eta = F.softplus(self.policy.vmpo_log_eta) + 1e-8  # type: ignore[attr-defined]
                # L_eta(eta) = eta * (epsilon_eta + log( (1/K) * sum_{i in S} exp(A_i / eta) ))
                # Use log-sum-exp: logmeanexp = logsumexp - logK
                logmeanexp = th.logsumexp(selected_adv_detached / eta, dim=0) - denom
                eta_loss_global = eta * (self.epsilon_eta + logmeanexp)

                self._update_learning_rate(self.eta_optimizer)
                self.eta_optimizer.zero_grad()
                eta_loss_global.backward()
                self.eta_optimizer.step()
                eta_losses.append(float(eta_loss_global.detach().cpu().item()))

        # Freeze eta for the whole policy update (all epochs in this iteration)
        eta_fixed = (F.softplus(self.policy.vmpo_log_eta) + 1e-8).detach()  # type: ignore[attr-defined]
        logsumexp_all_fixed = th.logsumexp(selected_adv_detached / eta_fixed, dim=0)

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                log_prob = log_prob.flatten()

                advantages = rollout_data.advantages
                batch_advantages.append(advantages.detach().cpu().numpy())

                advantages_estep_mb = advantages
                if self.normalize_advantage and advantages_estep_mb.numel() > 1:
                    if self.normalize_advantage_mean and adv_mean_t is not None:
                        advantages_estep_mb = advantages_estep_mb - adv_mean_t
                    if self.normalize_advantage_std and adv_std_t is not None:
                        advantages_estep_mb = advantages_estep_mb / adv_std_t
                advantages_estep_mb = advantages_estep_mb * self.advantage_multiplier

                mask_mb = advantages_estep_mb >= adv_threshold
                weights_mb = th.exp(advantages_estep_mb / eta_fixed - logsumexp_all_fixed) * mask_mb

                policy_loss = -(weights_mb * log_prob).sum()
                pg_losses.append(policy_loss.item())

                # --- V-MPO KL: state-averaged distribution KL (use reverse KL: KL(new || old)) ---
                with th.no_grad():
                    tgt_dist = self.target_policy.get_distribution(rollout_data.observations)  # old
                online_dist = self.policy.get_distribution(rollout_data.observations)  # new

                # want KL(new || old)
                kl_vec = _analytic_kl_sb3(online_dist, tgt_dist)  # [B]
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

                    alpha_m = F.softplus(self.policy.vmpo_log_alpha_mean) + 1e-8  # type: ignore[attr-defined]
                    alpha_s = F.softplus(self.policy.vmpo_log_alpha_std) + 1e-8  # type: ignore[attr-defined]

                    alpha_m_dual = alpha_m * (kl_mean.detach() - self.epsilon_alpha_mean)
                    alpha_s_dual = alpha_s * (kl_std.detach() - self.epsilon_alpha_std)
                    alpha_dual_mean.append(alpha_m_dual.item())
                    alpha_dual_std.append(alpha_s_dual.item())
                    alpha_losses.append((alpha_m_dual + alpha_s_dual).item())
                    kl_penalty = alpha_m.detach() * kl_mean + alpha_s.detach() * kl_std
                else:
                    alpha = F.softplus(self.policy.vmpo_log_alpha) + 1e-8  # type: ignore[attr-defined]
                    alpha_dual = alpha * (kl.detach() - self.epsilon_alpha)
                    alpha_losses.append(alpha_dual.item())
                    kl_penalty = alpha.detach() * kl
                # --- V-MPO KL constraint via alpha (Lagrange multiplier) ---
                # alpha = F.softplus(self.policy.vmpo_log_alpha) + 1e-8
                # alpha_dual = alpha * (kl.detach() - self.epsilon_alpha)
                # alpha_losses.append(alpha_dual.item())
                # kl_penalty = alpha.detach() * kl

                # Entropy loss (optional)
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Track weight entropy (diagnostic)
                with th.no_grad():
                    w = weights_mb
                    w_sum = w.sum().clamp_min(1e-12)
                    p = w / w_sum
                    w_ent = -(p * th.log(p.clamp_min(1e-12))).sum()
                    weight_entropies.append(w_ent.cpu().item())
                    weight_ess.append(th.exp(w_ent).cpu().item())

                # Critic loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                actor_total_loss = policy_loss + self.ent_coef * entropy_loss + kl_penalty
                critic_total_loss = self.vf_coef * value_loss
                total_loss = actor_total_loss + critic_total_loss

                # Optimize
                if self.separate_optimizers:
                    assert self.actor_optimizer is not None and self.critic_optimizer is not None
                    assert self._actor_params is not None and self._critic_params is not None

                    self._update_learning_rate(self.actor_optimizer)
                    self._update_learning_rate(self.critic_optimizer)

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    total_loss.backward()

                    actor_gn = th.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm).detach().cpu().numpy()
                    critic_gn = th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm).detach().cpu().numpy()
                    actor_grad_norms.append(actor_gn)
                    critic_grad_norms.append(critic_gn)

                    # Step both; shared feature extractor (if any) is only in critic optimizer.
                    self.critic_optimizer.step()
                    self.actor_optimizer.step()
                    # self._clamp_log_std()
                else:
                    self._update_learning_rate(self.policy.optimizer)
                    self.policy.optimizer.zero_grad()
                    total_loss.backward()
                    grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm).detach().cpu().numpy()
                    )
                    self.policy.optimizer.step()

                # Dual ascent step for alpha (separate from primal updates)
                if self.alpha_optimizer is not None:
                    self._update_learning_rate(self.alpha_optimizer)
                    self.alpha_optimizer.zero_grad()
                    if split_kl is not None:
                        alpha_m = F.softplus(self.policy.vmpo_log_alpha_mean) + 1e-8  # type: ignore[attr-defined]
                        alpha_s = F.softplus(self.policy.vmpo_log_alpha_std) + 1e-8  # type: ignore[attr-defined]
                        alpha_step_loss = -(
                            alpha_m * (kl_mean.detach() - self.epsilon_alpha_mean)
                            + alpha_s * (kl_std.detach() - self.epsilon_alpha_std)
                        )
                    else:
                        alpha = F.softplus(self.policy.vmpo_log_alpha) + 1e-8  # type: ignore[attr-defined]
                        alpha_step_loss = -(alpha * (kl.detach() - self.epsilon_alpha))
                    alpha_step_loss.backward()
                    self.alpha_optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

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

        _log_mean("train/policy_loss_vmpo", pg_losses)
        _log_mean("train/value_loss", value_losses)
        _log_mean("train/entropy_loss", entropy_losses)

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
        if hasattr(self.policy, "vmpo_log_eta"):
            eta_val = (F.softplus(self.policy.vmpo_log_eta) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
            self.logger.record("train/eta", eta_val)
        if hasattr(self.policy, "vmpo_log_alpha"):
            alpha_val = (F.softplus(self.policy.vmpo_log_alpha) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
            self.logger.record("train/alpha", alpha_val)
        if hasattr(self.policy, "vmpo_log_alpha_mean"):
            alpha_mean_val = (F.softplus(self.policy.vmpo_log_alpha_mean) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
            self.logger.record("train/alpha_mean", alpha_mean_val)
        if hasattr(self.policy, "vmpo_log_alpha_std"):
            alpha_std_val = (F.softplus(self.policy.vmpo_log_alpha_std) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
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
            return ["policy", "policy.optimizer", "actor_optimizer", "critic_optimizer", "alpha_optimizer", "eta_optimizer"], []
        # Default behavior from parent
        return super()._get_torch_save_params()

    def learn(
        self: SelfVMPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "VMPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfVMPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
