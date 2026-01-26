import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

import copy
import torch.distributions as torch_dist

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
    Compute per-state KL: D_KL(target || online) with best-effort support for SB3 dist wrappers.
    Returns: kl_per_state with shape [B]
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
        init_eta: float = 1.0,
        init_alpha: float = 1.0,
        top_adv_fraction: float = 0.5,
        # --- DAE (Direct Advantage Estimation) ---
        use_dae: bool = False,
        dae_coef: float = 1.0,
        dae_center_mc_samples: int = 8,
        dae_detach_policy_adv: bool = True,
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
        self.epsilon_eta = float(epsilon_eta)
        self.epsilon_alpha = float(epsilon_alpha)
        self.init_eta = float(init_eta)
        self.init_alpha = float(init_alpha)
        self.top_adv_fraction = float(top_adv_fraction)

        # Split-optimizer related attributes
        self.actor_optimizer: Optional[th.optim.Optimizer] = None
        self.critic_optimizer: Optional[th.optim.Optimizer] = None
        self._actor_params: Optional[list[th.nn.Parameter]] = None
        self._critic_params: Optional[list[th.nn.Parameter]] = None

        # Frozen snapshot used for the analytic distribution-KL constraint within each update
        self.target_policy: Optional[BasePolicy] = None

        # --- DAE flags/hparams ---
        self.use_dae = bool(use_dae)
        self.dae_coef = float(dae_coef)
        self.dae_center_mc_samples = int(dae_center_mc_samples)
        self.dae_detach_policy_adv = bool(dae_detach_policy_adv)
        assert self.dae_center_mc_samples >= 1, "`dae_center_mc_samples` must be >= 1"
        assert self.dae_coef >= 0.0, "`dae_coef` must be >= 0"
        self._last_dones: Optional[np.ndarray] = None  # <-- add (robustness)

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        """
        Store last-step dones for DAE bootstrapping mask.
        """
        ok = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        # SB3 convention: _last_episode_starts equals `dones` from the last env step.
        self._last_dones = np.array(self._last_episode_starts, copy=True)
        return ok

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

        # --- DAE: attach advantage head early so it's included in deepcopy(target_policy) and optimizers ---
        if self.use_dae:
            self._ensure_dae_adv_net()

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

            # Dual variables (temperature η and KL multiplier α) are part of the actor-side optimization.
            actor_params.append(self.policy.vmpo_log_eta)  # type: ignore[attr-defined]
            actor_params.append(self.policy.vmpo_log_alpha)  # type: ignore[attr-defined]

            # DAE advantage head: trained with DAE regression (place on critic-side by default)
            if self.use_dae and hasattr(self.policy, "dae_adv_net"):
                _extend_unique(critic_params, self.policy.dae_adv_net.parameters())  # type: ignore[attr-defined]

            self._actor_params = actor_params
            self._critic_params = critic_params

            initial_lr = self.lr_schedule(1)
            self.actor_optimizer = self.policy.optimizer_class(self._actor_params, lr=initial_lr, **self.policy.optimizer_kwargs)  # type: ignore[arg-type]
            self.critic_optimizer = self.policy.optimizer_class(self._critic_params, lr=initial_lr, **self.policy.optimizer_kwargs)  # type: ignore[arg-type]
        else:
            # Ensure dual params are optimized even with the single policy optimizer.
            self.policy.optimizer.add_param_group({"params": [self.policy.vmpo_log_eta, self.policy.vmpo_log_alpha]})  # type: ignore[attr-defined]
            if self.use_dae and hasattr(self.policy, "dae_adv_net"):
                self.policy.optimizer.add_param_group({"params": list(self.policy.dae_adv_net.parameters())})  # type: ignore[attr-defined]

        # Create target policy once; synced at the start of each train() call (per update)
        if self.target_policy is None:
            self.target_policy = copy.deepcopy(self.policy).to(self.device)
        self.target_policy.set_training_mode(False)

    def _ensure_dae_adv_net(self) -> None:
        """
        Attach a learned advantage head onto `self.policy`:
        - Discrete: A(s,·) in R^{|A|} from latent_pi
        - Box:      A(s,a) scalar via MLP([latent_pi, action])
        """
        if hasattr(self.policy, "dae_adv_net"):
            return

        if not isinstance(self.action_space, (spaces.Discrete, spaces.Box)):
            raise NotImplementedError("DAE is implemented for Discrete and Box action spaces only.")

        # Actor latent size (preferred); fallback for older SB3 variants
        latent_dim = int(getattr(self.policy.mlp_extractor, "latent_dim_pi", None) or self.policy.mlp_extractor.latent_pi_dim)  # type: ignore[attr-defined]

        if isinstance(self.action_space, spaces.Discrete):
            act_dim = int(self.action_space.n)
            self.policy.dae_adv_net = th.nn.Linear(latent_dim, act_dim).to(self.device)  # type: ignore[attr-defined]
        else:
            act_dim = int(np.prod(self.action_space.shape))
            self.policy._dae_act_dim = act_dim  # type: ignore[attr-defined]
            self.policy.dae_adv_net = th.nn.Sequential(  # type: ignore[attr-defined]
                th.nn.Linear(latent_dim + act_dim, 256),
                th.nn.Tanh(),
                th.nn.Linear(256, 1),
            ).to(self.device)

    def _dae_latent_pi(self, obs: Any) -> th.Tensor:
        """
        Actor latent used for A-hat. If policy uses separate feature extractors, use pi_features_extractor.
        """
        if getattr(self.policy, "share_features_extractor", True):
            features = self.policy.extract_features(obs)
        else:
            features = self.policy.extract_features(obs, features_extractor=self.policy.pi_features_extractor)
        latent_pi, _latent_vf = self.policy.mlp_extractor(features)
        return latent_pi

    def _dae_adv_centered(self, obs: Any, actions: th.Tensor) -> th.Tensor:
        """
        Centered learned advantage: A_hat(s,a) - E_{a~mu}[A_hat(s,a)], with mu = target_policy snapshot.
        Returned shape: [B]
        """
        self._ensure_dae_adv_net()
        assert self.target_policy is not None

        latent_pi = self._dae_latent_pi(obs)  # NOTE: no detach here; DAE may train representation

        if isinstance(self.action_space, spaces.Discrete):
            logits_A = self.policy.dae_adv_net(latent_pi)  # type: ignore[attr-defined]  # [B, act_dim]
            a = actions.long().view(-1, 1)
            A_sa = logits_A.gather(1, a).squeeze(1)

            with th.no_grad():
                mu = _sb3_base_distribution(self.target_policy.get_distribution(obs))
                mu_probs = mu.probs  # [B, act_dim]
            baseline = (mu_probs * logits_A).sum(dim=1)
            return A_sa - baseline

        act_dim = int(getattr(self.policy, "_dae_act_dim"))  # type: ignore[attr-defined]
        a = actions.view(-1, act_dim).float()
        A_sa = self.policy.dae_adv_net(th.cat([latent_pi, a], dim=1)).squeeze(1)  # type: ignore[attr-defined]

        with th.no_grad():
            mu = _sb3_base_distribution(self.target_policy.get_distribution(obs))
            K = int(self.dae_center_mc_samples)
            a_samp = mu.sample((K,))  # [K, B, act_dim] (best-effort)
            a_samp = a_samp.view(K * a.shape[0], act_dim).float()

        latent_rep = latent_pi.unsqueeze(0).expand(K, *latent_pi.shape).reshape(K * latent_pi.shape[0], -1)
        A_samp = self.policy.dae_adv_net(th.cat([latent_rep, a_samp], dim=1)).view(K, latent_pi.shape[0])  # type: ignore[attr-defined]
        baseline = A_samp.mean(dim=0)
        return A_sa - baseline

    def _dae_loss_on_rollout(self) -> th.Tensor:
        """
        DAE regression loss (Pan et al., Eq. 13) via backward recursion over the full rollout.
        Handles RolloutBuffer layouts that store obs/actions either as [T, N, ...] or [T*N, ...].
        """
        assert self._last_dones is not None, "DAE needs last-step dones; ensure collect_rollouts() was called."
        assert self.target_policy is not None

        self._ensure_dae_adv_net()

        # Determine (T, N) from episode_starts (most consistent across SB3 versions)
        episode_starts_np = self.rollout_buffer.episode_starts  # numpy
        if episode_starts_np.ndim != 2:
            raise ValueError(f"Expected rollout_buffer.episode_starts to be 2D [T,N], got shape={episode_starts_np.shape}")
        T, N = episode_starts_np.shape

        def _as_tn(x_np: np.ndarray, name: str) -> np.ndarray:
            """
            Ensure x is shaped [T, N, ...] (or [T, N] for scalars).
            Accepts already-shaped [T,N,...] or flattened [T*N,...].
            """
            if x_np.shape[0] == T and (x_np.ndim == 1 or x_np.shape[1] == N):
                # [T,N,...] or [T] (shouldn't happen for rollout tensors, but keep safe)
                return x_np if x_np.ndim != 1 else x_np.reshape(T, 1)
            if x_np.shape[0] == T * N:
                return x_np.reshape(T, N, *x_np.shape[1:])
            raise ValueError(
                f"Cannot coerce {name} to [T,N,...]. "
                f"T={T}, N={N}, got shape={x_np.shape} (expected first dim {T} or {T*N})."
            )

        # Rewards/episode_starts are expected [T,N] in SB3
        rewards = th.as_tensor(_as_tn(self.rollout_buffer.rewards, "rewards"), device=self.device).float()  # [T,N]
        episode_starts = th.as_tensor(episode_starts_np, device=self.device).float()  # [T,N]

        # Observations may be stored as [T,N,...] or flattened [T*N,...]
        obs_np = self.rollout_buffer.observations
        if isinstance(obs_np, dict):
            obs_tn = {k: _as_tn(v, f"observations[{k}]") for k, v in obs_np.items()}
            obs_flat = {k: th.as_tensor(v.reshape(T * N, *v.shape[2:]), device=self.device) for k, v in obs_tn.items()}
        else:
            obs_tn = _as_tn(obs_np, "observations")
            obs_flat = th.as_tensor(obs_tn.reshape(T * N, *obs_tn.shape[2:]), device=self.device)

        # Actions may be stored as [T,N,act_dim] or flattened [T*N,act_dim] (or [T,N] / [T*N] for discrete)
        actions_np = self.rollout_buffer.actions
        actions_tn = _as_tn(actions_np, "actions")

        if isinstance(self.action_space, spaces.Discrete):
            # SB3 often stores discrete actions with trailing dim=1; normalize to [T*N]
            act_flat = th.as_tensor(actions_tn.reshape(T * N, -1), device=self.device).long().squeeze(-1)
        else:
            act_flat = th.as_tensor(actions_tn.reshape(T * N, -1), device=self.device).float()

        # mask[t,n] = 1 if not terminal after step t
        mask = th.ones((T, N), device=self.device)
        if T > 1:
            mask[:-1] = 1.0 - episode_starts[1:]
        last_dones = th.as_tensor(self._last_dones, device=self.device).float()  # [N]
        mask[-1] = 1.0 - last_dones

        A = self._dae_adv_centered(obs_flat, act_flat).view(T, N)  # centered A-hat
        V = self.policy.predict_values(obs_flat).view(T, N)

        # Bootstrap V_target(s_T)
        last_obs_t, _ = self.policy.obs_to_tensor(self._last_obs)  # type: ignore[arg-type]
        with th.no_grad():
            V_T = self.target_policy.predict_values(last_obs_t).view(N) * (1.0 - last_dones)

        # Backward recursion
        y_next = V_T
        sqerrs = []
        for t in range(T - 1, -1, -1):
            y_t = (rewards[t] - A[t]) + self.gamma * mask[t] * y_next
            e_t = y_t - V[t]
            sqerrs.append((e_t ** 2).mean())
            y_next = y_t

        return th.stack(list(reversed(sqerrs))).mean()

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
        kls = []
        weight_entropies = []
        grad_norms = []
        actor_grad_norms: list[np.ndarray] = []
        critic_grad_norms: list[np.ndarray] = []
        batch_advantages = []
        batch_norm_advantages = []
        continue_training = True
        dae_losses = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                batch_advantages.append(advantages.detach().cpu().numpy())

                # --- Advantages used for VMPO E-step ---
                if self.use_dae:
                    adv_hat = self._dae_adv_centered(rollout_data.observations, actions)
                    advantages_estep = adv_hat.detach() if self.dae_detach_policy_adv else adv_hat
                else:
                    advantages_estep = advantages
                    if self.normalize_advantage and len(advantages_estep) > 1:
                        if self.normalize_advantage_mean:
                            advantages_estep = advantages_estep - advantages_estep.mean()
                            batch_norm_advantages.append(advantages_estep.detach().cpu().numpy())

                advantages_estep = advantages_estep * self.advantage_multiplier

                # --- V-MPO KL: analytic state-averaged distribution KL ---
                with th.no_grad():
                    tgt_dist = self.target_policy.get_distribution(rollout_data.observations)
                online_dist = self.policy.get_distribution(rollout_data.observations)
                kl_vec = _analytic_kl_sb3(tgt_dist, online_dist)  # [B]
                kl = kl_vec.mean()
                kls.append(kl.detach().cpu().item())
                approx_kl_divs.append(kl.detach().cpu().item())

                # Early stopping (optional)
                if self.target_kl is not None and kl.detach().cpu().item() > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max KL: {kl.detach().cpu().item():.4f}")
                    break

                # --- V-MPO E-step: select top-advantage samples and compute weights ---
                batch_n = advantages_estep.shape[0]
                k = max(1, int(self.top_adv_fraction * batch_n))
                topk = th.topk(advantages_estep, k=k, sorted=False)
                sel_adv = topk.values
                sel_log_prob = log_prob[topk.indices]

                eta = F.softplus(self.policy.vmpo_log_eta) + 1e-8  # type: ignore[attr-defined]
                logits = (sel_adv / eta.detach()).clamp(-50, 50)
                weights = th.softmax(logits, dim=0).detach()
                policy_loss = -(weights * sel_log_prob).sum()
                pg_losses.append(policy_loss.item())

                # --- V-MPO dual loss for eta (temperature) ---
                with th.no_grad():
                    denom = th.log(th.tensor(float(k), device=sel_adv.device, dtype=sel_adv.dtype))
                lse = th.logsumexp(sel_adv.detach() / eta, dim=0) - denom
                eta_loss = eta * (self.epsilon_eta + lse)
                eta_losses.append(eta_loss.item())

                # --- V-MPO KL constraint via alpha (Lagrange multiplier) ---
                alpha = F.softplus(self.policy.vmpo_log_alpha) + 1e-8  # type: ignore[attr-defined]
                alpha_loss = alpha * (self.epsilon_alpha - kl.detach())
                alpha_losses.append(alpha_loss.item())
                kl_penalty = alpha.detach() * kl

                # Entropy loss (optional)
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Track weight entropy (diagnostic)
                with th.no_grad():
                    w_ent = -(weights * th.log(weights.clamp_min(1e-12))).sum()
                    weight_entropies.append(w_ent.cpu().item())

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

                actor_total_loss = policy_loss + self.ent_coef * entropy_loss + kl_penalty + eta_loss + alpha_loss
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
                else:
                    self._update_learning_rate(self.policy.optimizer)
                    self.policy.optimizer.zero_grad()
                    total_loss.backward()
                    grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm).detach().cpu().numpy()
                    )
                    self.policy.optimizer.step()

            # --- DAE regression step (full-rollout, keeps temporal structure) ---
            if self.use_dae and self.dae_coef > 0.0:
                dae_loss = self._dae_loss_on_rollout()
                dae_losses.append(float(dae_loss.detach().cpu().item()))

                if self.separate_optimizers:
                    assert self.actor_optimizer is not None and self.critic_optimizer is not None
                    assert self._actor_params is not None and self._critic_params is not None

                    # DAE should be allowed to train representation (latent_pi path) + value path + dae_adv_net
                    self._update_learning_rate(self.actor_optimizer)
                    self._update_learning_rate(self.critic_optimizer)

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    (self.dae_coef * dae_loss).backward()

                    th.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm)
                    th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm)

                    self.critic_optimizer.step()
                    self.actor_optimizer.step()
                else:
                    self._update_learning_rate(self.policy.optimizer)
                    self.policy.optimizer.zero_grad()
                    (self.dae_coef * dae_loss).backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        if len(batch_advantages) > 0:
            self.logger.record("train/advantages_mean", np.mean(batch_advantages))
            self.logger.record("train/advantages_sum", np.sum(batch_advantages))
        if len(batch_norm_advantages) > 0:
            self.logger.record("train/advantages_norm_mean", np.mean(batch_norm_advantages))
            self.logger.record("train/advantages_norm_sum", np.sum(batch_norm_advantages))

        self.logger.record("train/policy_loss_vmpo", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/kl_pi_old_pi_new", np.mean(kls))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/eta_loss", np.mean(eta_losses))
        self.logger.record("train/alpha_loss", np.mean(alpha_losses))
        if len(weight_entropies) > 0:
            self.logger.record("train/weights_entropy", np.mean(weight_entropies))
        if len(dae_losses) > 0:
            self.logger.record("train/dae_loss", float(np.mean(dae_losses)))

        # Dual variable values
        if hasattr(self.policy, "vmpo_log_eta"):
            eta_val = (F.softplus(self.policy.vmpo_log_eta) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
            self.logger.record("train/eta", eta_val)
        if hasattr(self.policy, "vmpo_log_alpha"):
            alpha_val = (F.softplus(self.policy.vmpo_log_alpha) + 1e-8).detach().cpu().item()  # type: ignore[attr-defined]
            self.logger.record("train/alpha", alpha_val)

        self.logger.record("train/explained_variance", explained_var)

        if len(actor_grad_norms) > 0:
            self.logger.record("train/grad_norm_actor", float(np.mean(actor_grad_norms)))
            self.logger.record("train/grad_norm_actor/max", float(np.max(actor_grad_norms)))
        if len(critic_grad_norms) > 0:
            self.logger.record("train/grad_norm_critic", float(np.mean(critic_grad_norms)))
            self.logger.record("train/grad_norm_critic/max", float(np.max(critic_grad_norms)))
        if len(grad_norms) > 0:
            self.logger.record("train/grad_norm", float(np.mean(grad_norms)))
            self.logger.record("train/grad_norm/max", float(np.max(grad_norms)))

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Include separate optimizers in state dicts when enabled.
        """
        if self.separate_optimizers:
            return ["policy", "policy.optimizer", "actor_optimizer", "critic_optimizer"], []
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
