"""Continual Learning, Elastic Weight Consolidation, and Rehearsal Memory for Oxideon.

Enables learning consecutive tasks without catastrophic forgetting of previous knowledge:
  - ElasticWeightConsolidation (EWC): Fisher Information quadratic penalty regularizing critical weights (Kirkpatrick et al., 2017)
  - estimate_fisher_information: Empirical diagonal Fisher information calculator
  - RehearsalMemory: Dynamic experience replay reservoir preserving representative task exemplars
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


def _iter_named_params(model: Any) -> List[Tuple[str, Any]]:
    np_fn = getattr(model, "named_parameters", None)
    if callable(np_fn):
        res = list(np_fn())
        if res:
            return res
    params_dict = getattr(model, "_parameters", None)
    if isinstance(params_dict, dict) and params_dict:
        out: List[Tuple[str, Any]] = []
        for k, v in params_dict.items():
            if v is not None:
                out.append((k, v))
        modules_dict = getattr(model, "_modules", None)
        if isinstance(modules_dict, dict):
            for m_name, mod in modules_dict.items():
                if mod is not None:
                    for sub_k, sub_v in _iter_named_params(mod):
                        out.append((f"{m_name}.{sub_k}", sub_v))
        if out:
            return out
    # Native PyO3 layers (e.g. Linear, Conv2d, Embedding, LayerNorm) expose .weight and .bias
    direct: List[Tuple[str, Any]] = []
    for attr in ("weight", "bias", "gamma", "beta"):
        val = getattr(model, attr, None)
        if isinstance(val, _Tensor):
            direct.append((attr, val))
    if direct:
        return direct
    p_fn = getattr(model, "parameters", None)
    if callable(p_fn):
        return [(f"param_{i}", p) for i, p in enumerate(p_fn())]
    return []


class ElasticWeightConsolidation(Module):
    """Elastic Weight Consolidation (EWC) penalty regularizing critical parameters of past tasks."""

    def __init__(
        self,
        model: Module,
        fisher_dict: Dict[str, np.ndarray],
        optimal_params: Dict[str, np.ndarray],
        importance: float = 100.0,
    ):
        super().__init__()
        if float(importance) < 0.0:
            raise ValueError(f"importance must be >= 0, got {importance}")
        self.model = model
        self.fisher = fisher_dict
        self.optimal_params = optimal_params
        self.importance = float(importance)

    def forward(self) -> _Tensor:
        """Calculates differentiable quadratic penalty: sum_i (importance / 2) * F_i * (theta_i - theta_opt_i)^2."""
        total: Optional[_Tensor] = None
        for name, param in _iter_named_params(self.model):
            if name in self.fisher and name in self.optimal_params:
                f_np = np.asarray(
                    self.fisher[name].to_numpy() if isinstance(self.fisher[name], _Tensor) else self.fisher[name],
                    dtype=np.float32,
                )
                opt_np = np.asarray(
                    self.optimal_params[name].to_numpy()
                    if isinstance(self.optimal_params[name], _Tensor)
                    else self.optimal_params[name],
                    dtype=np.float32,
                )
                f_t = _from_numpy(np.ascontiguousarray(f_np))
                opt_t = _from_numpy(np.ascontiguousarray(opt_np))
                diff = param - opt_t
                term = (f_t * (diff * diff)).sum()
                total = term if total is None else (total + term)

        if total is None:
            return _from_numpy(np.array([0.0], dtype=np.float32))
        return (total * (self.importance / 2.0)).reshape([1])


def estimate_fisher_information(
    model: Module,
    samples: List[Tuple[_Tensor, _Tensor]],
    loss_fn: Any,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Estimates diagonal Fisher Information matrix and records optimal parameters."""
    if len(samples) == 0:
        raise ValueError("samples cannot be empty when estimating Fisher Information")
    named_params = _iter_named_params(model)
    fisher = {name: np.zeros_like(p.to_numpy()) for name, p in named_params}
    optimal = {name: p.to_numpy().copy() for name, p in named_params}

    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()
    n = len(samples)

    for x, y in samples:
        for _, p in named_params:
            if hasattr(p, "zero_grad"):
                p.zero_grad()
            elif hasattr(p, "grad"):
                try:
                    p.grad = None
                except AttributeError:
                    pass

        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()

        for name, param in named_params:
            g_call = getattr(param, "grad", None)
            g = g_call() if callable(g_call) else g_call
            if g is not None:
                g_arr = g.to_numpy()
                fisher[name] += (g_arr**2) / max(n, 1)

    return fisher, optimal


class RehearsalMemory:
    """Reservoir sampling memory buffer storing past task exemplars for replay."""

    def __init__(self, capacity: int = 100, seed: Optional[int] = None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.buffer_x: List[np.ndarray] = []
        self.buffer_y: List[np.ndarray] = []
        self.total_seen = 0
        self._rng = np.random.default_rng(seed)

    def add(self, x: Any, y: Any):
        """Adds sample to buffer using reservoir sampling."""
        x_arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float32)
        y_arr = y.to_numpy() if isinstance(y, _Tensor) else np.asarray(y)
        if self.buffer_x and x_arr.shape != self.buffer_x[0].shape:
            raise ValueError(f"Sample x shape {x_arr.shape} does not match buffer shape {self.buffer_x[0].shape}")
        if self.buffer_y and y_arr.shape != self.buffer_y[0].shape:
            raise ValueError(f"Sample y shape {y_arr.shape} does not match buffer shape {self.buffer_y[0].shape}")

        self.total_seen += 1
        if len(self.buffer_x) < self.capacity:
            self.buffer_x.append(x_arr)
            self.buffer_y.append(y_arr)
        else:
            j = int(self._rng.integers(0, self.total_seen))
            if j < self.capacity:
                self.buffer_x[j] = x_arr
                self.buffer_y[j] = y_arr

    def sample(self, batch_size: int) -> Tuple[_Tensor, _Tensor]:
        """Draws uniform replay mini-batch from rehearsal buffer."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        n = len(self.buffer_x)
        if n == 0:
            raise RuntimeError("Rehearsal buffer is empty.")

        k = min(batch_size, n)
        idx = self._rng.choice(n, size=k, replace=False)
        batch_x = np.stack([self.buffer_x[i] for i in idx], axis=0)
        batch_y = np.stack([self.buffer_y[i] for i in idx], axis=0)

        return (
            _from_numpy(np.ascontiguousarray(batch_x, dtype=np.float32)),
            _from_numpy(np.ascontiguousarray(batch_y)),
        )

    def clear(self) -> None:
        """Clears all stored exemplars and resets total_seen counter."""
        self.buffer_x.clear()
        self.buffer_y.clear()
        self.total_seen = 0

    def __len__(self) -> int:
        return len(self.buffer_x)


__all__ = [
    "ElasticWeightConsolidation",
    "estimate_fisher_information",
    "RehearsalMemory",
]
