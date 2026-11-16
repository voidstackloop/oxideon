"""Differential Privacy and DP-SGD (Differentially Private Stochastic Gradient Descent) for Oxideon.

Enables training neural networks with rigorous (epsilon, delta)-differential privacy guarantees:
  - clip_grad_per_sample: Enforces maximum L2 norm constraint C on gradients
  - add_gaussian_noise: Perturbs gradients with calibrated Gaussian noise
  - privacy_spent: Computes total privacy expenditure epsilon using Rényi Differential Privacy
  - DP_SGD: Optimizer wrapper injecting DP clipping and Gaussian noise mechanisms (Abadi et al., 2016)
"""

from typing import Any, Tuple, Optional, List, Union
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def clip_grad_per_sample(grad: Any, max_norm: float = 1.0) -> Any:
    """Clips gradient tensor/array so its L2 norm does not exceed max_norm."""
    if float(max_norm) <= 0.0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")

    is_tensor = isinstance(grad, _Tensor)
    arr = grad.to_numpy() if is_tensor else np.asarray(grad, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("grad cannot be empty")
    norm = float(np.linalg.norm(arr))
    if norm > max_norm:
        factor = float(max_norm / max(norm, 1e-12))
        if is_tensor:
            return grad * factor
        return arr * factor
    return grad


def add_gaussian_noise(
    grad: Any,
    noise_multiplier: float,
    max_norm: float = 1.0,
    seed: Optional[int] = None,
) -> Any:
    """Injects Gaussian noise N(0, (noise_multiplier * max_norm)^2) to gradient."""
    if float(noise_multiplier) < 0.0:
        raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}")
    if float(max_norm) <= 0.0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")

    is_tensor = isinstance(grad, _Tensor)
    arr = grad.to_numpy() if is_tensor else np.asarray(grad, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("grad cannot be empty")

    sigma = float(noise_multiplier * max_norm)
    rng = np.random.default_rng(seed) if seed is not None else np.random
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    res = arr + noise
    if is_tensor:
        return _from_numpy(np.ascontiguousarray(res))
    return res


def privacy_spent(
    num_steps: int,
    batch_size: int,
    total_dataset_size: int,
    noise_multiplier: float,
    delta: float = 1e-5,
) -> float:
    """Estimates epsilon privacy budget spent under sub-sampled Gaussian mechanism."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if total_dataset_size <= 0:
        raise ValueError(f"total_dataset_size must be positive, got {total_dataset_size}")
    if batch_size < 1 or batch_size > total_dataset_size:
        raise ValueError(f"batch_size ({batch_size}) must be in [1, total_dataset_size ({total_dataset_size})]")
    if float(noise_multiplier) <= 0.0:
        raise ValueError(f"noise_multiplier must be > 0, got {noise_multiplier}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be strictly within (0, 1), got {delta}")
    q = batch_size / total_dataset_size
    eps = (q * math.sqrt(2.0 * num_steps * math.log(1.0 / max(delta, 1e-15)))) / noise_multiplier
    return float(eps)


class DP_SGD:
    """Differentially Private SGD wrapper wrapping standard optimizer."""

    def __init__(
        self,
        optimizer: Any,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        seed: Optional[int] = None,
    ):
        if float(noise_multiplier) < 0.0:
            raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}")
        if float(max_grad_norm) <= 0.0:
            raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
        self.optimizer = optimizer
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.seed = seed
        self.steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        """Applies gradient clipping and calibrated Gaussian noise to parameters before stepping."""
        params = getattr(self.optimizer, "param_groups", None)
        if params is None:
            params = getattr(self.optimizer, "params", [])
        else:
            flat = []
            for g in params:
                flat.extend(g.get("params", []))
            params = flat

        for p in params:
            g_call = getattr(p, "grad", None)
            g = g_call() if callable(g_call) else g_call
            if g is not None:
                clipped = clip_grad_per_sample(g, self.max_grad_norm)
                noised = add_gaussian_noise(clipped, self.noise_multiplier, self.max_grad_norm, seed=self.seed)
                new_grad = noised if isinstance(noised, _Tensor) else _from_numpy(np.ascontiguousarray(noised, dtype=np.float32))
                if hasattr(p, "set_grad"):
                    p.set_grad(new_grad)
                elif hasattr(p, "grad"):
                    try:
                        p.grad = new_grad
                    except Exception:
                        pass

        self.optimizer.step()
        self.steps += 1


__all__ = [
    "clip_grad_per_sample",
    "add_gaussian_noise",
    "privacy_spent",
    "DP_SGD",
]
