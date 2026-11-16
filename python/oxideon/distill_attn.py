"""Attention Map Transfer and Hidden State Distillation for Oxideon.

Enables distilling spatial attention maps and intermediate representations from massive Transformers:
  - AttentionMapDistillationLoss: Attention transfer via normalized spatial activation maps (Zagoruyko & Komodakis, 2016)
  - HiddenStatesDistillationLoss: Matches hidden representations between student and teacher layers
  - AttentionHeadPruner: Evaluates and masks redundant attention heads based on energy
"""

from typing import Tuple, Optional, Any, List
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Parameter


class AttentionMapDistillationLoss(Module):
    """Zagoruyko & Komodakis (2016) Attention Transfer loss with Tensor autograd."""

    def __init__(self, p: int = 2):
        super().__init__()
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        self.p = p

    def forward(
        self, student_attn: Any, teacher_attn: Any
    ) -> _Tensor:
        """Takes student and teacher attention maps [B, H, T, T], [B, T, T], or [B, C, H, W] and computes L2 normalized gap."""
        s_t = student_attn if isinstance(student_attn, _Tensor) else _from_numpy(np.ascontiguousarray(student_attn, dtype=np.float32))
        t_t = teacher_attn if isinstance(teacher_attn, _Tensor) else _from_numpy(np.ascontiguousarray(teacher_attn, dtype=np.float32))

        s_pow = s_t.abs().pow(float(self.p))
        t_pow = t_t.detach().abs().pow(float(self.p))

        s_map = s_pow.sum(dim=1) if s_t.ndim == 4 else s_pow
        t_map = t_pow.sum(dim=1) if t_t.ndim == 4 else t_pow

        if list(s_map.shape) != list(t_map.shape):
            raise ValueError(
                f"Shape mismatch after channel/head collapse: student {tuple(s_map.shape)} != teacher {tuple(t_map.shape)}"
            )

        b = s_map.shape[0]
        feat_dim = int(np.prod(s_map.shape[1:]))
        s_flat = s_map.reshape([b, feat_dim])
        t_flat = t_map.reshape([b, feat_dim])

        s_denom = ((s_flat * s_flat).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
        t_denom = ((t_flat * t_flat).sum(dim=-1, keepdim=True) + 1e-12).sqrt()

        s_norm = s_flat / s_denom
        t_norm = t_flat / t_denom

        diff = s_norm - t_norm
        loss = (diff * diff).sum(dim=-1).mean()
        return loss.reshape([1])


class HiddenStatesDistillationLoss(Module):
    """Matches hidden representations across student and teacher layers with linear projection."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        if student_dim < 1 or teacher_dim < 1:
            raise ValueError("student_dim and teacher_dim must be >= 1")
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim

        if student_dim != teacher_dim:
            self.proj = Linear(student_dim, teacher_dim)
        else:
            self.proj = None

    def forward(self, student_h: _Tensor, teacher_h: _Tensor) -> _Tensor:
        h_s = self.proj(student_h) if self.proj is not None else student_h
        diff = h_s - teacher_h.detach()
        return (diff * diff).mean()


class AttentionHeadPruner:
    """Evaluates attention head importance by L1 energy and generates head masks."""

    def __init__(self, num_heads: int):
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        self.num_heads = num_heads
        self.head_mask = np.ones(num_heads, dtype=np.float32)

    def prune_heads(self, head_attentions: Any, amount: float = 0.25) -> np.ndarray:
        """Prunes lowest energy heads: head_attentions has shape [B, num_heads, T, T] or [num_heads, T, T]."""
        if not (0.0 <= amount <= 1.0):
            raise ValueError(f"amount must be in [0, 1], got {amount}")
        arr = head_attentions.to_numpy() if isinstance(head_attentions, _Tensor) else np.asarray(head_attentions)
        if arr.ndim == 3:
            if arr.shape[0] != self.num_heads:
                raise ValueError(f"Expected num_heads={self.num_heads} at dim 0, got {arr.shape[0]}")
            energies = np.sum(np.abs(arr), axis=(1, 2))
        elif arr.ndim == 4:
            if arr.shape[1] != self.num_heads:
                raise ValueError(f"Expected num_heads={self.num_heads} at dim 1, got {arr.shape[1]}")
            energies = np.sum(np.abs(arr), axis=(0, 2, 3))
        else:
            raise ValueError(f"Expected 3D or 4D attention tensor, got {arr.ndim}D")

        k = int(self.num_heads * amount)
        if k > 0:
            least_important = np.argsort(energies)[:k]
            self.head_mask[least_important] = 0.0

        return self.head_mask


__all__ = [
    "AttentionMapDistillationLoss",
    "HiddenStatesDistillationLoss",
    "AttentionHeadPruner",
]
