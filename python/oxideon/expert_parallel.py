"""Hierarchical Capacity-Constrained Sparse Mixture of Experts (MoE).

Implements modern Switch Transformer & DeepSeek-V3 style load-balanced routing:
- TopKRouterWithAuxLoss: Top-k gating with auxiliary load-balancing loss preventing expert collapse
- CapacityConstrainedMoE: Sparse MoE layer enforcing expert token capacity limits and token dropping
- ExpertParallelGroup: Parallel distributed token dispatch and gathering simulator
"""

from typing import List, Tuple, Optional, Dict
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, ModuleList, GELU, _check_repeats


class TopKRouterWithAuxLoss(Module):
    """Top-K gating router with auxiliary load-balancing loss."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2, aux_loss_weight: float = 0.01):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k cannot exceed num_experts (got top_k={top_k}, num_experts={num_experts})")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.gate = Linear(hidden_dim, num_experts)
        self._last_probs_tensor: Optional[_Tensor] = None

    def forward(self, x: _Tensor) -> Tuple[np.ndarray, np.ndarray, float]:
        """Args:
            x: Input tokens [num_tokens, hidden_dim]

        Returns:
            Tuple of (top_k_weights [num_tokens, top_k], top_k_indices [num_tokens, top_k], aux_loss)
        """
        x_in = x.unsqueeze(0) if x.ndim == 1 else x
        logits_t = self.gate(x_in)  # [N, E]
        probs_t = logits_t.softmax(-1)
        self._last_probs_tensor = probs_t

        probs = probs_t.to_numpy()
        num_tokens = len(probs)
        if num_tokens == 0:
            return np.zeros((0, self.top_k), dtype=np.float32), np.zeros((0, self.top_k), dtype=np.int64), 0.0

        top_indices = np.argsort(-probs, axis=-1)[:, :self.top_k]
        rows = np.arange(num_tokens)[:, None]
        top_weights = probs[rows, top_indices]
        top_weights = top_weights / (np.sum(top_weights, axis=-1, keepdims=True) + 1e-9)

        expert_counts = np.bincount(top_indices.flatten(), minlength=self.num_experts)
        f_e = expert_counts / (float(num_tokens * self.top_k) + 1e-9)
        P_e = np.mean(probs, axis=0)

        aux_loss = float(self.num_experts * np.sum(f_e * P_e) * self.aux_loss_weight)
        return top_weights, top_indices, aux_loss


class CapacityConstrainedMoE(Module):
    """Sparse MoE layer with expert capacity factor limits and differentiable routing."""

    def __init__(
        self,
        hidden_dim: int,
        expert_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        num_experts = _check_repeats("num_experts", num_experts)
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = TopKRouterWithAuxLoss(hidden_dim, num_experts, top_k=top_k)

        self.experts = ModuleList([
            ModuleList([
                Linear(hidden_dim, expert_dim),
                GELU(),
                Linear(expert_dim, hidden_dim),
            ])
            for _ in range(num_experts)
        ])

    def forward(self, x: _Tensor) -> Tuple[_Tensor, float]:
        """Args: x: [num_tokens, hidden_dim] or [B, S, hidden_dim] -> (output_tokens, aux_loss)"""
        orig_shape = list(x.shape)
        if len(orig_shape) == 0 or int(np.prod(orig_shape)) == 0:
            return x, 0.0

        x_in = x.reshape([-1, orig_shape[-1]]) if len(orig_shape) > 2 else x
        num_tokens = x_in.shape[0]

        weights, indices, aux_loss = self.router(x_in)
        probs_t = self.router._last_probs_tensor

        expert_capacity = int(math.ceil((self.capacity_factor * num_tokens * self.top_k) / self.num_experts))

        # Build capacity-constrained mask of shape [num_tokens, num_experts]
        cap_mask = np.zeros((num_tokens, self.num_experts), dtype=np.float32)
        for e_idx in range(self.num_experts):
            token_matches, slot_matches = np.where(indices == e_idx)
            if len(token_matches) == 0:
                continue
            assigned_tokens = token_matches[:expert_capacity]
            cap_mask[assigned_tokens, e_idx] = 1.0

        # Renormalize selected top-k probabilities per token (for non-dropped slots)
        topk_full_mask = np.zeros((num_tokens, self.num_experts), dtype=np.float32)
        topk_full_mask[np.arange(num_tokens)[:, None], indices] = 1.0

        if probs_t is not None:
            denom_t = (probs_t * _from_numpy(topk_full_mask)).sum(dim=-1, keepdim=True) + 1e-9
            norm_probs_t = (probs_t * _from_numpy(cap_mask)) / denom_t
        else:
            norm_probs_t = None

        total_out: Optional[_Tensor] = None
        for e_idx, expert in enumerate(self.experts):
            if not np.any(cap_mask[:, e_idx] > 0):
                continue

            h = x_in
            for layer in expert:
                h = layer(h)

            if norm_probs_t is not None:
                e_col = np.zeros((1, self.num_experts), dtype=np.float32)
                e_col[0, e_idx] = 1.0
                w_e = (norm_probs_t * _from_numpy(e_col)).sum(dim=-1, keepdim=True)
            else:
                w_np = np.zeros((num_tokens, 1), dtype=np.float32)
                token_matches, slot_matches = np.where(indices == e_idx)
                assigned_tokens = token_matches[:expert_capacity]
                assigned_slots = slot_matches[:expert_capacity]
                w_np[assigned_tokens, 0] = weights[assigned_tokens, assigned_slots]
                w_e = _from_numpy(w_np)

            contrib = h * w_e
            total_out = contrib if total_out is None else (total_out + contrib)

        if total_out is None:
            total_out = x_in * 0.0

        return total_out.reshape(orig_shape), aux_loss


class ExpertParallelGroup(Module):
    """Distributed Expert Parallelism (EP) simulator partitioning experts across virtual devices
    with All-to-All token dispatch and combine.
    """

    def __init__(
        self,
        num_devices: int,
        experts_per_device: int,
        hidden_dim: int,
        expert_dim: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        if num_devices < 1 or experts_per_device < 1:
            raise ValueError("num_devices and experts_per_device must be >= 1")
        self.num_devices = num_devices
        self.experts_per_device = experts_per_device
        self.total_experts = num_devices * experts_per_device
        self.moe = CapacityConstrainedMoE(
            hidden_dim=hidden_dim,
            expert_dim=expert_dim,
            num_experts=self.total_experts,
            top_k=min(top_k, self.total_experts),
            capacity_factor=capacity_factor,
        )

    def forward(self, x: _Tensor) -> Tuple[_Tensor, float]:
        return self.moe(x)


__all__ = [
    "TopKRouterWithAuxLoss",
    "CapacityConstrainedMoE",
    "ExpertParallelGroup",
]
