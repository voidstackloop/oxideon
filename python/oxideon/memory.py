"""Memory-Augmented Neural Architectures and Associative Memory for Oxideon.

Enables persistent episodic state, long-context caching, and differentiable retrieval:
  - KVMemoryBank: Differentiable Key-Value Memory Network (Sukhbaatar et al., 2015)
  - RingAttentionContext: Block-wise chunked attention context manager (Liu et al., 2023)
  - EpisodicMemory: Nearest-neighbor associative memory for episodic RL / LLM agents (Pritzel et al., 2017)
"""

from typing import Tuple, Optional, Any, List
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .functional import softmax
from .nn import Module, Parameter


class KVMemoryBank(Module):
    """Differentiable Key-Value Memory Bank with soft addressing and content-based reading."""

    def __init__(self, dim: int, capacity: int = 128):
        super().__init__()
        if dim < 1 or capacity < 1:
            raise ValueError(f"dim and capacity must be >= 1, got dim={dim}, capacity={capacity}")
        self.dim = dim
        self.capacity = capacity

        # Memory matrices
        self.keys = np.zeros((capacity, dim), dtype=np.float32)
        self.values = np.zeros((capacity, dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return self.size

    def write(self, keys: Any, values: Any):
        """Writes key-value pairs into memory slots."""
        k = keys.to_numpy() if isinstance(keys, _Tensor) else np.asarray(keys, dtype=np.float32)
        v = values.to_numpy() if isinstance(values, _Tensor) else np.asarray(values, dtype=np.float32)
        if k.ndim == 1:
            k = k[None, :]
        if v.ndim == 1:
            v = v[None, :]
        if k.shape != v.shape:
            raise ValueError(f"Shape mismatch between keys {k.shape} and values {v.shape}")
        if k.shape[-1] != self.dim:
            raise ValueError(f"Key/value dimension {k.shape[-1]} must match memory dim {self.dim}")

        n = len(k)
        for i in range(n):
            idx = (self.ptr + i) % self.capacity
            self.keys[idx] = k[i]
            self.values[idx] = v[i]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def read(self, query: Any, temperature: float = 1.0) -> _Tensor:
        """Reads from memory via soft dot-product attention over stored keys."""
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        q = query.to_numpy() if isinstance(query, _Tensor) else np.asarray(query, dtype=np.float32)
        is_1d = q.ndim == 1
        if is_1d:
            q = q[None, :]
        elif q.ndim != 2:
            raise ValueError(f"query must be 1D [D] or 2D [B, D], got shape {q.shape}")
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dimension {q.shape[1]} must match memory dim {self.dim}")

        if self.size == 0:
            zeros = np.zeros((q.shape[0], self.dim), dtype=np.float32)
            base = _from_numpy(zeros.squeeze(0) if is_1d else zeros)
            if isinstance(query, _Tensor) and query.requires_grad:
                # Zeros carry no information, but keep gradient flow so
                # backward() works (grads are correctly zero).
                q_t = query.reshape([1, self.dim]) if is_1d else query
                return base + (q_t.sum() * 0.0)
            return base

        active_keys = self.keys[: self.size]
        active_values = self.values[: self.size]

        if isinstance(query, _Tensor) and query.requires_grad:
            q_t = query.reshape([1, self.dim]) if is_1d else query
            k_t = _from_numpy(np.ascontiguousarray(active_keys.T, dtype=np.float32))
            v_t = _from_numpy(np.ascontiguousarray(active_values, dtype=np.float32))
            scores = q_t.matmul(k_t) * (1.0 / (math.sqrt(self.dim) * float(temperature)))
            weights = scores.softmax(-1)
            out_t = weights.matmul(v_t)
            return out_t.reshape([self.dim]) if is_1d else out_t

        # Scaled dot-product: [B, D] @ [size, D].T -> [B, size]
        scores = (q @ active_keys.T) / (math.sqrt(self.dim) * temperature)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        # Weighted sum: [B, size] @ [size, D] -> [B, D]
        out = weights @ active_values
        if is_1d:
            out = out.squeeze(0)
        return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))

    def clear(self):
        self.size = 0
        self.ptr = 0


class RingAttentionContext:
    """Simulates block-wise ring-attention across sequence chunks."""

    def __init__(self, chunk_size: int = 128):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.chunk_size = chunk_size

    def chunk_sequence(self, sequence: Any) -> List[_Tensor]:
        """Divides long sequence [B, T, D] or [T, D] into manageable temporal chunks."""
        if isinstance(sequence, _Tensor) and sequence.requires_grad:
            ndim = len(sequence.shape)
            if ndim == 2:
                t = sequence.shape[0]
                return [
                    sequence.narrow(0, i, min(self.chunk_size, t - i))
                    for i in range(0, t, self.chunk_size)
                ]
            elif ndim == 3:
                t = sequence.shape[1]
                return [
                    sequence.narrow(1, i, min(self.chunk_size, t - i))
                    for i in range(0, t, self.chunk_size)
                ]

        arr = sequence.to_numpy() if isinstance(sequence, _Tensor) else np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 2:
            t, d = arr.shape
            return [
                _from_numpy(np.ascontiguousarray(arr[i : i + self.chunk_size, :], dtype=np.float32))
                for i in range(0, t, self.chunk_size)
            ]
        elif arr.ndim == 3:
            b, t, d = arr.shape
            return [
                _from_numpy(np.ascontiguousarray(arr[:, i : i + self.chunk_size, :], dtype=np.float32))
                for i in range(0, t, self.chunk_size)
            ]
        else:
            raise ValueError(f"sequence must be 2D [T, D] or 3D [B, T, D], got shape {arr.shape}")


class EpisodicMemory:
    """Non-parametric episodic memory buffer storing state-value pairs."""

    def __init__(self, state_dim: int, value_dim: int, capacity: int = 100):
        if state_dim < 1 or value_dim < 1 or capacity < 1:
            raise ValueError("state_dim, value_dim, and capacity must be >= 1")
        self.state_dim = state_dim
        self.value_dim = value_dim
        self.capacity = capacity

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.values = np.zeros((capacity, value_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state: Any, value: Any):
        s = state.to_numpy().reshape(-1) if isinstance(state, _Tensor) else np.asarray(state, dtype=np.float32).reshape(-1)
        v = value.to_numpy().reshape(-1) if isinstance(value, _Tensor) else np.asarray(value, dtype=np.float32).reshape(-1)
        if len(s) != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected state_dim={self.state_dim}, got {len(s)}")
        if len(v) != self.value_dim:
            raise ValueError(f"Value dimension mismatch: expected value_dim={self.value_dim}, got {len(v)}")

        self.states[self.ptr] = s
        self.values[self.ptr] = v
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def query_nearest(self, query_state: Any, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Finds k nearest memory states by L2 distance and returns their stored values."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        q = query_state.to_numpy().reshape(-1) if isinstance(query_state, _Tensor) else np.asarray(query_state, dtype=np.float32).reshape(-1)
        if len(q) != self.state_dim:
            raise ValueError(f"Query state dimension mismatch: expected state_dim={self.state_dim}, got {len(q)}")
        if self.size == 0:
            return np.empty((0, self.state_dim)), np.empty((0, self.value_dim))

        active_states = self.states[: self.size]
        dists = np.sum((active_states - q) ** 2, axis=-1)

        k_adj = min(k, self.size)
        indices = np.argsort(dists)[:k_adj]
        return self.states[indices], self.values[indices]


__all__ = [
    "KVMemoryBank",
    "RingAttentionContext",
    "EpisodicMemory",
]
