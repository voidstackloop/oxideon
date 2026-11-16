"""Model merging in weight space: average, task arithmetic, TIES, DARE, SLERP.

All mergers operate on state dicts (``{name: Tensor}``) with identical keys
and shapes, return fresh dicts, and never mutate their inputs:

- ``average_merge``: (weighted) mean of full weights.
- ``task_arithmetic_merge``: base plus weighted sum of task vectors.
- ``ties_merge``: trim low-magnitude deltas, elect per-param signs, disjoint
  mean over agreeing models (Yadav et al., 2024).
- ``dare_merge``: random delta dropout with rescaling, then mean (Yu et al., 2024).
- ``slerp_merge``: spherical interpolation between two models (with lerp
  fallback for zero or colinear tensors, and orthogonal-plane rotation for antiparallel tensors).
"""

from typing import Any, Dict, List, Sequence

import math

import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _as_state(state: Dict[str, _Tensor], name: str) -> Dict[str, np.ndarray]:
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{name} must be a non-empty state dict")
    out = {}
    for key, value in state.items():
        if not isinstance(value, _Tensor):
            raise ValueError(f"{name}['{key}'] must be a Tensor")
        arr = np.asarray(value.to_numpy(), dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}['{key}'] must be finite")
        out[key] = arr
    return out


def _check_compatible(base: Dict[str, np.ndarray], others: Sequence[Dict[str, np.ndarray]]) -> List[str]:
    keys = sorted(base)
    for i, other in enumerate(others):
        if sorted(other) != keys:
            missing = sorted(set(keys) - set(other))
            extra = sorted(set(other) - set(keys))
            raise ValueError(f"state {i} key mismatch (missing={missing}, extra={extra})")
        for key in keys:
            if other[key].shape != base[key].shape:
                raise ValueError(
                    f"state {i}['{key}'] shape {other[key].shape} != {base[key].shape}"
                )
    return keys


def _as_weights(weights: Any, n: int, name: str) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64).flatten()
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} entries, expected {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _pack(states: Dict[str, np.ndarray]) -> Dict[str, _Tensor]:
    return {k: _from_numpy(np.ascontiguousarray(v.astype(np.float32))) for k, v in states.items()}


def average_merge(
    states: Sequence[Dict[str, _Tensor]], weights: Any = None
) -> Dict[str, _Tensor]:
    """Normalized weighted mean of full model weights."""
    states = list(states)
    if not states:
        raise ValueError("average_merge needs at least one state")
    parsed = [_as_state(s, f"states[{i}]") for i, s in enumerate(states)]
    keys = _check_compatible(parsed[0], parsed[1:])
    w = _as_weights(weights, len(parsed), "weights")
    if w.sum() == 0:
        raise ValueError("weights must not sum to zero")
    w = w / w.sum()
    merged = {k: sum(w[i] * parsed[i][k] for i in range(len(parsed))) for k in keys}
    return _pack(merged)


def task_arithmetic_merge(
    base: Dict[str, _Tensor],
    states: Sequence[Dict[str, _Tensor]],
    weights: Any = None,
) -> Dict[str, _Tensor]:
    """Base plus weighted task vectors; default weights average the deltas."""
    states = list(states)
    if not states:
        raise ValueError("task_arithmetic_merge needs at least one finetune")
    base_np = _as_state(base, "base")
    parsed = [_as_state(s, f"states[{i}]") for i, s in enumerate(states)]
    keys = _check_compatible(base_np, parsed)
    if weights is None:
        w = np.ones(len(parsed), dtype=np.float64) / len(parsed)
    else:
        w = _as_weights(weights, len(parsed), "weights")
    merged = {
        k: base_np[k] + sum(w[i] * (parsed[i][k] - base_np[k]) for i in range(len(parsed)))
        for k in keys
    }
    return _pack(merged)


def ties_merge(
    base: Dict[str, _Tensor],
    states: Sequence[Dict[str, _Tensor]],
    weights: Any = None,
    density: float = 1.0,
) -> Dict[str, _Tensor]:
    """TIES merging: trim, elect signs, disjoint mean (per tensor)."""
    states = list(states)
    if not states:
        raise ValueError("ties_merge needs at least one finetune")
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density = {density} must sit within (0, 1]")
    base_np = _as_state(base, "base")
    parsed = [_as_state(s, f"states[{i}]") for i, s in enumerate(states)]
    keys = _check_compatible(base_np, parsed)
    w = _as_weights(weights, len(parsed), "weights")
    merged = {}
    for key in keys:
        deltas = np.stack([w[i] * (parsed[i][key] - base_np[key]) for i in range(len(parsed))])
        flat = deltas.reshape(len(parsed), -1)
        keep = int(density * flat.shape[1])
        if keep < flat.shape[1]:
            order = np.argsort(-np.abs(flat), axis=1, kind="stable")
            mask = np.zeros_like(flat, dtype=bool)
            rows = np.arange(flat.shape[0])[:, None]
            mask[rows, order[:, :keep]] = True
            flat = np.where(mask, flat, 0.0)
        total = flat.sum(axis=0)
        elected = np.where(total >= 0, 1.0, -1.0)
        agree = (np.sign(flat) == elected[None, :]) & (flat != 0.0)
        summed = (flat * agree).sum(axis=0)
        counts = agree.sum(axis=0)
        mean = np.divide(summed, counts, out=np.zeros_like(summed), where=counts > 0)
        merged[key] = base_np[key] + mean.reshape(base_np[key].shape)
    return _pack(merged)


def dare_merge(
    base: Dict[str, _Tensor],
    states: Sequence[Dict[str, _Tensor]],
    weights: Any = None,
    drop_rate: float = 0.9,
    seed: int = 0,
) -> Dict[str, _Tensor]:
    """DARE merging: drop deltas at ``drop_rate``, rescale survivors, mean."""
    states = list(states)
    if not states:
        raise ValueError("dare_merge needs at least one finetune")
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"drop_rate = {drop_rate} must sit within [0, 1)")
    if not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
        raise ValueError(f"seed = {seed} must be an integer within [0, 2**32)")
    base_np = _as_state(base, "base")
    parsed = [_as_state(s, f"states[{i}]") for i, s in enumerate(states)]
    keys = _check_compatible(base_np, parsed)
    w = _as_weights(weights, len(parsed), "weights")
    if w.sum() == 0:
        raise ValueError("weights must not sum to zero")
    w = w / w.sum()
    rng = np.random.default_rng(int(seed))
    keep_prob = 1.0 - drop_rate
    merged = {}
    for key in keys:
        acc = np.zeros_like(base_np[key])
        for i in range(len(parsed)):
            delta = parsed[i][key] - base_np[key]
            mask = rng.random(delta.shape) >= drop_rate
            acc = acc + w[i] * (delta * mask / keep_prob)
        merged[key] = base_np[key] + acc
    return _pack(merged)


def slerp_merge(
    state_a: Dict[str, _Tensor], state_b: Dict[str, _Tensor], t: float = 0.5
) -> Dict[str, _Tensor]:
    """Spherical interpolation per tensor; exact endpoints, lerp/orthogonal fallback."""
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t = {t} must sit within [0, 1]")
    left = _as_state(state_a, "state_a")
    right = _as_state(state_b, "state_b")
    keys = _check_compatible(left, [right])
    if t == 0.0:
        return _pack({k: left[k].copy() for k in keys})
    if t == 1.0:
        return _pack({k: right[k].copy() for k in keys})
    merged = {}
    for key in keys:
        a = left[key].flatten()
        b = right[key].flatten()
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            merged[key] = ((1.0 - t) * a + t * b).reshape(left[key].shape)
            continue
        cos_theta = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        theta = math.acos(cos_theta)
        sin_theta = math.sin(theta)
        if theta < 1e-6:
            merged[key] = ((1.0 - t) * a + t * b).reshape(left[key].shape)
            continue
        if abs(math.pi - theta) < 1e-6 or sin_theta < 1e-6:
            # Antiparallel vectors: rotate in an orthogonal 2D plane if dim >= 2, else lerp
            if len(a) >= 2:
                u = a / na
                # Pick a standard basis vector not collinear with u
                e = np.zeros_like(u)
                min_idx = int(np.argmin(np.abs(u)))
                e[min_idx] = 1.0
                v_orth = e - np.dot(e, u) * u
                v_orth = v_orth / (np.linalg.norm(v_orth) + 1e-12)
                interp_norm = (1.0 - t) * na + t * nb
                vec = interp_norm * (math.cos(t * math.pi) * u + math.sin(t * math.pi) * v_orth)
                merged[key] = vec.reshape(left[key].shape)
            else:
                merged[key] = ((1.0 - t) * a + t * b).reshape(left[key].shape)
            continue
        wa = math.sin((1.0 - t) * theta) / sin_theta
        wb = math.sin(t * theta) / sin_theta
        merged[key] = (wa * a + wb * b).reshape(left[key].shape)
    return _pack(merged)


__all__ = [
    "average_merge",
    "task_arithmetic_merge",
    "ties_merge",
    "dare_merge",
    "slerp_merge",
]
