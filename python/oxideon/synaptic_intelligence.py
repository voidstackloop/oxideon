"""Synaptic Intelligence and Experience Replay for Continual Learning.

Implements Continual Learning without Catastrophic Forgetting (Zenke et al., ICML 2017):
- PathIntegralRegularizer: Computes running gradient-displacement path integral to measure synaptic importance Omega
- SynapticIntelligenceTrainer: Coordinates sequential multi-task learning with quadratic parameter preservation
- MemoryReplayBuffer: Reservoir sampling experience replay buffer
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


def _iter_model_named_params(model: Module) -> List[Tuple[str, _Tensor]]:
    named = list(model.named_parameters()) if hasattr(model, "named_parameters") else []
    if named:
        return named
    if hasattr(model, "parameters"):
        return [(f"param_{i}", p) for i, p in enumerate(model.parameters())]
    return []


class PathIntegralRegularizer(Module):
    """Accumulates parameter importance Omega along continuous optimization trajectories."""

    def __init__(self, model: Module, c: float = 1.0, xi: float = 0.1):
        super().__init__()
        self.model = model
        self.c = float(c)
        self.xi = float(xi)

        self.omega_accum: Dict[str, np.ndarray] = {}
        self.prev_params: Dict[str, np.ndarray] = {}
        self.big_omega: Dict[str, np.ndarray] = {}
        self.reference_params: Dict[str, np.ndarray] = {}

        self.reset_trajectory()

    def reset_trajectory(self):
        """Initializes parameter snapshots at the beginning of a new task."""
        self.omega_accum = {}
        self.prev_params = {}
        for name, param in _iter_model_named_params(self.model):
            p_np = param.to_numpy().copy()
            self.prev_params[name] = p_np
            self.omega_accum[name] = np.zeros_like(p_np)

    def on_step(self):
        """Updates small omega running sum after an optimizer step."""
        for name, param in _iter_model_named_params(self.model):
            curr_p = param.to_numpy()
            if name not in self.prev_params:
                self.prev_params[name] = curr_p.copy()
                self.omega_accum[name] = np.zeros_like(curr_p)
            delta_p = curr_p - self.prev_params[name]
            grad_obj = param.grad() if callable(param.grad) else param.grad
            if grad_obj is not None:
                g_np = grad_obj.to_numpy()
                self.omega_accum[name] -= g_np * delta_p
            self.prev_params[name] = curr_p.copy()

    def consolidate_task(self):
        """Computes big Omega importance at the conclusion of a task."""
        for name, param in _iter_model_named_params(self.model):
            curr_p = param.to_numpy()
            delta_theta = curr_p - self.reference_params.get(name, self.prev_params.get(name, curr_p))
            omega = np.maximum(0.0, self.omega_accum.get(name, np.zeros_like(curr_p)))
            importance = omega / (delta_theta ** 2 + self.xi)

            if name in self.big_omega:
                self.big_omega[name] += importance
            else:
                self.big_omega[name] = importance.copy()

            self.reference_params[name] = curr_p.copy()

        self.reset_trajectory()

    def penalty_loss(self) -> _Tensor:
        """Computes differentiable quadratic surrogate penalty L = c * sum_k Omega_k * (theta_k - theta_k^*)^2."""
        total_penalty: Optional[_Tensor] = None
        for name, param in _iter_model_named_params(self.model):
            if name in self.big_omega and name in self.reference_params:
                ref_t = _from_numpy(np.ascontiguousarray(self.reference_params[name].astype(np.float32)))
                omega_t = _from_numpy(np.ascontiguousarray(self.big_omega[name].astype(np.float32)))
                diff = param - ref_t
                term = (diff * diff * omega_t).sum()
                total_penalty = term if total_penalty is None else (total_penalty + term)

        if total_penalty is None:
            return _from_numpy(np.array([0.0], dtype=np.float32))
        return total_penalty * float(self.c)

    surrogate_loss = penalty_loss


class MemoryReplayBuffer:
    """Reservoir sampling memory buffer preserving past task exemplars with uniform probability."""

    def __init__(self, capacity: int = 500):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.samples: List[Tuple[np.ndarray, Any]] = []
        self._total_seen = 0

    def add(self, x: np.ndarray, y: Any):
        """Adds a sample using reservoir sampling to maintain uniform distribution."""
        self._total_seen += 1
        if len(self.samples) < self.capacity:
            self.samples.append((x.copy(), y))
        else:
            idx = np.random.randint(0, self._total_seen)
            if idx < self.capacity:
                self.samples[idx] = (x.copy(), y)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Draws a random uniform mini-batch of past exemplars."""
        if len(self.samples) == 0:
            raise RuntimeError("Memory buffer is empty.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        n = min(batch_size, len(self.samples))
        indices = np.random.choice(len(self.samples), size=n, replace=False)
        xs = [self.samples[i][0] for i in indices]
        ys = [self.samples[i][1] for i in indices]
        return np.array(xs, dtype=np.float32), np.array(ys)


__all__ = [
    "PathIntegralRegularizer",
    "MemoryReplayBuffer",
]
