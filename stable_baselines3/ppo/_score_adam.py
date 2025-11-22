import math
from typing import Iterable, Optional

import torch as th


class ScoreAdam(th.optim.Optimizer):
    """
    Adam-like optimizer where the adaptive denominator (second-moment estimate)
    can be computed using gradients of the score function (∇ log π) instead of
    the full policy gradient that includes the advantage.

    The idea is:
    - Numerator (first moment / momentum) still uses the true gradient of the
      policy loss, which contains the advantage term.
    - Denominator (second moment) can optionally use an alternative gradient
      `score_grad` that only depends on the score function.

    When used with PPO in this codebase, we pass in score-only gradients
    computed from `log_prob.mean()` so that the variance estimate in the
    denominator does not depend on the advantage.
    """

    def __init__(
        self,
        params: Iterable[th.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @th.no_grad()
    def step(self, closure: Optional[callable] = None, score_grads: Optional[dict[int, th.Tensor]] = None):
        """
        Perform a single optimization step.

        :param closure: Optional closure that re-evaluates the model and returns the loss.
        :param score_grads: Optional dict mapping id(param) -> score-only gradient tensor.
            If provided, the second-moment (denominator) will use these gradients instead
            of the full gradient. If a param is missing, its normal gradient is used.
        """
        loss = None
        if closure is not None:
            with th.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ScoreAdam does not support sparse gradients")

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = th.zeros_like(p, memory_format=th.preserve_format)
                    # Exponential moving average of squared score gradients
                    state["exp_avg_sq"] = th.zeros_like(p, memory_format=th.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp_avg_sq
                        state["max_exp_avg_sq"] = th.zeros_like(p, memory_format=th.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Use true gradient (includes advantage) for first moment
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # For the denominator, optionally use score-only gradient
                score_grad = grad
                if score_grads is not None:
                    maybe_sg = score_grads.get(id(p))
                    if maybe_sg is not None:
                        score_grad = maybe_sg

                exp_avg_sq.mul_(beta2).addcmul_(score_grad, score_grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    # Maintains the maximum of all 2nd moment running averages
                    th.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt()).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt()).add_(eps)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


