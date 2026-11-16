"""Knowledge editing and machine unlearning for post-training updates.

- ``rome_update``: rank-one model edit solving W' k* = v* (Meng et al., 2023).
- ``memit_layer_update``: batched multi-key edit with covariance prior and
  residual spreading across layers (Meng et al., 2023).
- ``NPOLoss``: Negative Preference Optimization unlearning from forget-set
  logps against a reference (Zhang et al., 2024).
- ``GALoss``: gradient-ascent unlearning with optional retain regularization.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module


def _vec(values: Any, name: str, n: int) -> np.ndarray:
    arr = np.asarray(
        values.to_numpy() if isinstance(values, _Tensor) else values, dtype=np.float64
    ).flatten()
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} entries, expected {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _mat(values: Any, name: str, rows: int, cols: int) -> np.ndarray:
    arr = np.asarray(
        values.to_numpy() if isinstance(values, _Tensor) else values, dtype=np.float64
    )
    if arr.shape != (rows, cols):
        raise ValueError(f"{name} shape {arr.shape} != expected ({rows}, {cols})")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def rome_update(
    weight: _Tensor, key: Any, value_star: Any, covariance: Any
) -> _Tensor:
    """Rank-one edit: returns W' with W' k* = v* and minimal change elsewhere.

    Solves u = C^-1 k*, then W' = W + (v* - W k*) u^T / (u^T k*).
    """
    w_np = np.asarray(weight.to_numpy(), dtype=np.float64)
    if w_np.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {w_np.shape}")
    if not np.all(np.isfinite(w_np)):
        raise ValueError("weight must be finite")
    out_dim, in_dim = w_np.shape
    k = _vec(key, "key", in_dim)
    v = _vec(value_star, "value_star", out_dim)
    cov = _mat(covariance, "covariance", in_dim, in_dim)
    try:
        u = np.linalg.solve(cov, k)
    except np.linalg.LinAlgError:
        raise ValueError("covariance is singular; add damping or use a full-rank estimate")
    denom = float(u @ k)
    if abs(denom) < 1e-12:
        raise ValueError("key lies in the null space of the covariance")
    resid = v - w_np @ k
    edited = w_np + np.outer(resid, u) / denom
    return _from_numpy(np.ascontiguousarray(edited.astype(np.float32)))


def memit_layer_update(
    weight: _Tensor,
    keys: Any,
    values_star: Any,
    covariance: Any,
    layers_remaining: int = 1,
    damping: float = 0.0,
) -> _Tensor:
    """Batched MEMIT update: W' = W + R K^T (C + K K^T + dI)^-1."""
    w_np = np.asarray(weight.to_numpy(), dtype=np.float64)
    if w_np.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {w_np.shape}")
    if not np.all(np.isfinite(w_np)):
        raise ValueError("weight must be finite")
    out_dim, in_dim = w_np.shape
    keys_np = np.asarray(
        keys.to_numpy() if isinstance(keys, _Tensor) else keys, dtype=np.float64
    )
    vals_np = np.asarray(
        values_star.to_numpy() if isinstance(values_star, _Tensor) else values_star,
        dtype=np.float64,
    )
    if keys_np.ndim != 2 or keys_np.shape[0] < 1 or keys_np.shape[1] != in_dim:
        raise ValueError(f"keys must be [N, {in_dim}] with N >= 1")
    if vals_np.shape != (keys_np.shape[0], out_dim):
        raise ValueError(f"values_star shape {vals_np.shape} != ({keys_np.shape[0]}, {out_dim})")
    if not np.all(np.isfinite(keys_np)) or not np.all(np.isfinite(vals_np)):
        raise ValueError("keys and values_star must be finite")
    cov = _mat(covariance, "covariance", in_dim, in_dim)
    if not isinstance(layers_remaining, (int, np.integer)) or layers_remaining < 1:
        raise ValueError("layers_remaining must be an integer >= 1")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    keys_t = keys_np.T
    vals_t = vals_np.T
    resid = (vals_t - w_np @ keys_t) / layers_remaining
    mat = cov + keys_t @ keys_t.T
    if damping:
        mat = mat + damping * np.eye(in_dim)
    try:
        solved = np.linalg.solve(mat, keys_t @ resid.T)
    except np.linalg.LinAlgError:
        raise ValueError("normal matrix is singular; increase damping")
    edited = w_np + solved.T
    return _from_numpy(np.ascontiguousarray(edited.astype(np.float32)))


class NPOLoss(Module):
    """Negative Preference Optimization unlearning (Zhang et al., 2024).

    L = -mean(log sigmoid(-beta * (logp - logp_ref))) over forget samples:
    pushes the policy below the reference exactly where it memorized.
    Preserves Tensor autograd w.r.t. `policy_logps`.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)

    def forward(
        self, policy_logps: _Tensor, reference_logps: _Tensor
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """Unlearning loss over flat forget-set batches."""
        p_raw = policy_logps.to_numpy() if hasattr(policy_logps, "to_numpy") else np.asarray(policy_logps)
        r_raw = reference_logps.to_numpy() if hasattr(reference_logps, "to_numpy") else np.asarray(reference_logps)
        if p_raw.shape != r_raw.shape:
            raise ValueError(f"Shape mismatch: policy_logps {p_raw.shape} != reference_logps {r_raw.shape}")
        logp = np.asarray(p_raw, dtype=np.float64).flatten()
        ref = np.asarray(r_raw, dtype=np.float64).flatten()
        if logp.shape != ref.shape or logp.shape[0] < 1:
            raise ValueError("policy/reference logps must be non-empty flat batches of equal size")
        if not np.all(np.isfinite(logp)) or not np.all(np.isfinite(ref)):
            raise ValueError("policy/reference logps must be finite")

        p_t = policy_logps if isinstance(policy_logps, _Tensor) else _from_numpy(np.ascontiguousarray(p_raw.astype(np.float32)))
        r_t = reference_logps if isinstance(reference_logps, _Tensor) else _from_numpy(np.ascontiguousarray(r_raw.astype(np.float32)))

        z_t = (p_t - r_t) * (-self.beta)
        loss_t = (z_t.sigmoid() + 1e-12).log().mean() * (-1.0)
        loss_val = float(loss_t.to_numpy().item())
        metrics = {
            "npo_loss": loss_val,
            "mean_log_ratio": float(np.mean(logp - ref)),
        }
        return loss_t.reshape([1]), metrics


class GALoss(Module):
    """Gradient-ascent unlearning with optional retain regularization.

    Minimizes mean forget logps (descending memorized likelihood) while
    maximizing retain logps when provided. Preserves Tensor autograd.
    """

    def __init__(self, retain_weight: float = 1.0):
        super().__init__()
        if retain_weight < 0:
            raise ValueError("retain_weight must be non-negative")
        self.retain_weight = float(retain_weight)

    def forward(
        self, forget_logps: _Tensor, retain_logps: Optional[_Tensor] = None
    ) -> Tuple[_Tensor, Dict[str, float]]:
        """mean(logp_forget) - w * mean(logp_retain)."""
        f_arr = forget_logps.to_numpy() if hasattr(forget_logps, "to_numpy") else np.asarray(forget_logps)
        forget = np.asarray(f_arr, dtype=np.float64).flatten()
        if forget.shape[0] < 1 or not np.all(np.isfinite(forget)):
            raise ValueError("forget_logps must be a non-empty finite batch")

        f_t = forget_logps if isinstance(forget_logps, _Tensor) else _from_numpy(np.ascontiguousarray(f_arr.astype(np.float32)))
        loss_t = f_t.mean()
        mean_forget = float(np.mean(forget))
        metrics: Dict[str, float] = {"ga_loss": mean_forget, "mean_forget_logp": mean_forget}

        if retain_logps is not None:
            r_arr = retain_logps.to_numpy() if hasattr(retain_logps, "to_numpy") else np.asarray(retain_logps)
            retain = np.asarray(r_arr, dtype=np.float64).flatten()
            if retain.shape[0] < 1 or not np.all(np.isfinite(retain)):
                raise ValueError("retain_logps must be a non-empty finite batch")
            r_t = retain_logps if isinstance(retain_logps, _Tensor) else _from_numpy(np.ascontiguousarray(r_arr.astype(np.float32)))
            loss_t = loss_t - r_t.mean() * self.retain_weight
            mean_retain = float(np.mean(retain))
            metrics["ga_loss"] = float(loss_t.to_numpy().item())
            metrics["mean_retain_logp"] = mean_retain

        return loss_t.reshape([1]), metrics


__all__ = [
    "rome_update",
    "memit_layer_update",
    "NPOLoss",
    "GALoss",
]
