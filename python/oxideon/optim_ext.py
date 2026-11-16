"""Advanced Frontier Optimizers and Learning Rate Schedulers.

Implements:
17. ShampooOptimizer: Matrix preconditioned 2nd-order optimizer
18. AdEMAMix: Fast and slow dual exponential moving average optimizer (Apple 2024)
19. ProdigyOptimizer: Adaptive learning-rate-free D-adaptation optimizer
20. SophiaOptimizer: Second-order curvature-clipped optimizer (Liu et al., 2023)
21. ScheduleFreeSGD: Schedule-free optimization without learning rate decay
22. LookaheadOptimizer: Slow-fast weight interpolation optimizer wrapper
23. CosineAnnealingWarmRestarts (SGDRScheduler): Stochastic gradient descent with warm restarts
"""

from typing import Dict, List, Optional, Any, Tuple
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Parameter


def _get_grad(p) -> Optional[np.ndarray]:
    g = p.grad() if callable(p.grad) else p.grad
    return g.to_numpy() if g is not None else None


def _update_param(p, new_array: np.ndarray) -> None:
    t = _from_numpy(np.ascontiguousarray(new_array.astype(np.float32)))
    if hasattr(p, "assign"):
        p.assign(t)
    elif hasattr(p, "copy_"):
        p.copy_(t)
    elif hasattr(p, "_inner"):
        p._inner = t._inner


def _zero_grad_param(p) -> None:
    if hasattr(p, "zero_grad"):
        p.zero_grad()
    elif hasattr(p, "set_grad"):
        p.set_grad(None)


class ShampooOptimizer:
    """Matrix preconditioned 2nd-order optimizer (Gupta et al., 2018)."""

    def __init__(self, params: List[Parameter], lr: float = 1e-3, beta2: float = 0.999, epsilon: float = 1e-4):
        self.params = params
        self.lr = lr
        self.beta2 = beta2
        self.epsilon = epsilon
        self.left_precond: Dict[int, np.ndarray] = {}
        self.right_precond: Dict[int, np.ndarray] = {}

    def step(self):
        for p in self.params:
            g = _get_grad(p)
            if g is None:
                continue
            p_id = id(p)

            if g.ndim == 2:
                l_stat = g @ g.T
                r_stat = g.T @ g

                if p_id not in self.left_precond:
                    self.left_precond[p_id] = l_stat
                    self.right_precond[p_id] = r_stat
                else:
                    self.left_precond[p_id] = self.beta2 * self.left_precond[p_id] + (1 - self.beta2) * l_stat
                    self.right_precond[p_id] = self.beta2 * self.right_precond[p_id] + (1 - self.beta2) * r_stat

                inv_l = 1.0 / (np.diag(self.left_precond[p_id]) ** 0.25 + self.epsilon)[:, None]
                inv_r = 1.0 / (np.diag(self.right_precond[p_id]) ** 0.25 + self.epsilon)[None, :]
                preconditioned_g = g * inv_l * inv_r
            else:
                preconditioned_g = g

            new_p = p.to_numpy() - self.lr * preconditioned_g
            _update_param(p, new_p)

    def zero_grad(self):
        for p in self.params:
            _zero_grad_param(p)


class AdEMAMix:
    """AdEMAMix: Fast and Slow Dual Exponential Moving Average Optimizer (Pagliardini et al., Apple 2024)."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.9999,
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.m_fast: Dict[int, np.ndarray] = {}
        self.m_slow: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.step_count = 0

    def step(self):
        self.step_count += 1
        for p in self.params:
            g = _get_grad(p)
            if g is None:
                continue
            p_id = id(p)

            if p_id not in self.m_fast:
                self.m_fast[p_id] = np.zeros_like(g)
                self.m_slow[p_id] = np.zeros_like(g)
                self.v[p_id] = np.zeros_like(g)

            self.m_fast[p_id] = self.beta1 * self.m_fast[p_id] + (1.0 - self.beta1) * g
            self.m_slow[p_id] = self.beta3 * self.m_slow[p_id] + (1.0 - self.beta3) * g
            self.v[p_id] = self.beta2 * self.v[p_id] + (1.0 - self.beta2) * (g ** 2)

            m_combined = self.m_fast[p_id] + self.alpha * self.m_slow[p_id]
            denom = np.sqrt(self.v[p_id]) + self.eps

            curr_p = p.to_numpy()
            update = m_combined / denom + self.weight_decay * curr_p
            new_p = curr_p - self.lr * update
            _update_param(p, new_p)

    def zero_grad(self):
        for p in self.params:
            _zero_grad_param(p)


class ProdigyOptimizer:
    """Prodigy: Adaptive Learning-Rate-Free Optimizer via D-Adaptation (Mishchenko & Defazio, 2024)."""

    def __init__(self, params: List[Parameter], lr: float = 1.0, d_init: float = 1e-4, weight_decay: float = 0.0, beta1: float = 0.9, beta2: float = 0.999):
        self.params = params
        self.lr = lr
        self.d = float(d_init)
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.s: Dict[int, np.ndarray] = {}
        self.x0: Dict[int, np.ndarray] = {}
        self.d_numerator: float = 0.0

    def step(self):
        sqrt_b2 = math.sqrt(self.beta2)
        self.d_numerator *= sqrt_b2
        d_denom = 0.0

        for p in self.params:
            g = _get_grad(p)
            if g is None:
                continue
            p_id = id(p)
            curr_p = p.to_numpy()

            if p_id not in self.m:
                self.m[p_id] = np.zeros_like(g)
                self.v[p_id] = np.zeros_like(g)
                self.s[p_id] = np.zeros_like(g)
                self.x0[p_id] = curr_p.copy()

            dlr = self.lr * self.d
            self.m[p_id] = self.beta1 * self.m[p_id] + (1.0 - self.beta1) * self.d * g
            self.v[p_id] = self.beta2 * self.v[p_id] + (1.0 - self.beta2) * (self.d ** 2) * (g ** 2)
            self.s[p_id] = sqrt_b2 * self.s[p_id] + (1.0 - sqrt_b2) * dlr * g

            self.d_numerator += float(dlr * np.sum(g * (self.x0[p_id] - curr_p)))
            d_denom += float(np.sum(np.abs(self.s[p_id]) / (np.sqrt(self.v[p_id]) + 1e-8)))

            update = self.m[p_id] / (np.sqrt(self.v[p_id]) + self.d * 1e-8)
            if self.weight_decay > 0.0:
                update = update + self.weight_decay * curr_p
            new_p = curr_p - dlr * update
            _update_param(p, new_p)

        if d_denom > 1e-12:
            d_hat = self.d_numerator / ((1.0 - sqrt_b2) * d_denom)
            self.d = max(self.d, float(d_hat))

    def zero_grad(self):
        for p in self.params:
            _zero_grad_param(p)


class SophiaOptimizer:
    """Sophia: Second-order Stochastic Curvature-Clipped Optimizer (Liu et al., 2023)."""

    def __init__(self, params: List[Parameter], lr: float = 1e-3, beta1: float = 0.96, beta2: float = 0.99, rho: float = 0.04):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.m: Dict[int, np.ndarray] = {}
        self.h: Dict[int, np.ndarray] = {}

    def update_hessian(self, hessian_dict: Dict[int, np.ndarray]):
        """Updates diagonal Hessian curvature estimates."""
        for p_id, h in hessian_dict.items():
            self.h[p_id] = h.copy()

    def step(self):
        for p in self.params:
            g = _get_grad(p)
            if g is None:
                continue
            p_id = id(p)

            if p_id not in self.m:
                self.m[p_id] = np.zeros_like(g)
                self.h[p_id] = np.zeros_like(g)

            self.m[p_id] = self.beta1 * self.m[p_id] + (1.0 - self.beta1) * g
            self.h[p_id] = self.beta2 * self.h[p_id] + (1.0 - self.beta2) * (g ** 2)

            ratio = self.m[p_id] / (self.h[p_id] + 1e-8)
            clipped = np.clip(ratio, -self.rho, self.rho)

            curr_p = p.to_numpy()
            new_p = curr_p - self.lr * clipped
            _update_param(p, new_p)

    def zero_grad(self):
        for p in self.params:
            _zero_grad_param(p)


class ScheduleFreeSGD:
    """Schedule-Free SGD Optimizer without Learning Rate Decay (Defazio et al., 2024)."""

    def __init__(self, params: List[Parameter], lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_count = 0
        self.x_bar: Dict[int, np.ndarray] = {}
        self.z: Dict[int, np.ndarray] = {}

    def step(self):
        self.step_count += 1
        c = self.step_count
        for p in self.params:
            g = _get_grad(p)
            if g is None:
                continue
            p_id = id(p)

            curr_p = p.to_numpy()
            if p_id not in self.x_bar:
                self.x_bar[p_id] = curr_p.copy()
                self.z[p_id] = curr_p.copy()

            if self.weight_decay > 0.0:
                g = g + self.weight_decay * curr_p

            self.z[p_id] -= self.lr * g
            self.x_bar[p_id] = (1.0 - 1.0 / c) * self.x_bar[p_id] + (1.0 / c) * self.z[p_id]
            y_interp = (1.0 - self.momentum) * self.z[p_id] + self.momentum * self.x_bar[p_id]

            _update_param(p, y_interp if self.momentum < 1.0 else self.x_bar[p_id])

    def zero_grad(self):
        for p in self.params:
            _zero_grad_param(p)


class LookaheadOptimizer:
    """Lookahead: k-step slow-fast weight interpolation wrapper (Zhang et al., NeurIPS 2019)."""

    def __init__(self, base_optimizer: Any, k: int = 5, alpha: float = 0.5):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = {
            id(p): p.to_numpy().copy() for p in base_optimizer.params
        }

    def step(self):
        self.base_optimizer.step()
        self.step_count += 1

        if self.step_count % self.k == 0:
            for p in self.base_optimizer.params:
                slow = self.slow_weights[id(p)]
                fast = p.to_numpy()
                new_slow = slow + self.alpha * (fast - slow)
                self.slow_weights[id(p)] = new_slow.copy()
                _update_param(p, new_slow)

    def zero_grad(self):
        self.base_optimizer.zero_grad()


class SGDRScheduler:
    """SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, ICLR 2017)."""

    def __init__(self, optimizer: Any, t_0: int = 10, t_mult: int = 2, eta_min: float = 0.0):
        if t_0 < 1:
            raise ValueError(f"t_0 must be >= 1, got {t_0}")
        if t_mult < 1:
            raise ValueError(f"t_mult must be >= 1, got {t_mult}")
        self.optimizer = optimizer
        self.t_0 = t_0
        self.t_mult = t_mult
        self.eta_min = eta_min
        self.base_lr = optimizer.lr if hasattr(optimizer, "lr") else 1e-3
        self.t_cur = 0
        self.t_i = t_0

    def step(self):
        self.t_cur += 1
        if self.t_cur >= self.t_i:
            self.t_cur = 0
            self.t_i = self.t_i * self.t_mult

        cos_val = math.cos(math.pi * self.t_cur / self.t_i)
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + cos_val)

        if hasattr(self.optimizer, "lr"):
            self.optimizer.lr = lr
        elif hasattr(self.optimizer, "set_lr"):
            self.optimizer.set_lr(lr)


from .optim.lr_scheduler import CosineAnnealingWarmRestarts


__all__ = [
    "ShampooOptimizer",
    "AdEMAMix",
    "ProdigyOptimizer",
    "SophiaOptimizer",
    "ScheduleFreeSGD",
    "LookaheadOptimizer",
    "SGDRScheduler",
    "CosineAnnealingWarmRestarts",
]
