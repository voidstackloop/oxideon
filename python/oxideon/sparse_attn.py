"""
Sparse Attention mechanisms and long-context sub-quadratic transformers for Oxideon.
Includes BigBird attention (Zaheer et al.), sliding window (banded) attention, and global token routing.
"""
from typing import Optional
import math
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, _check_elements


def build_sliding_window_mask(seq_len: int, window_radius: int) -> np.ndarray:
    """Creates a binary sliding window attention mask of shape (seq_len, seq_len)."""
    seq_len = _check_elements("seq_len", seq_len)
    if window_radius < 0:
        raise ValueError(f"window_radius must be non-negative (>= 0), got {window_radius}")
    idx = np.arange(seq_len)
    dist = np.abs(idx[:, None] - idx[None, :])
    return (dist <= window_radius).astype(np.float32)


def build_bigbird_mask(
    seq_len: int,
    num_global: int = 2,
    window_radius: int = 3,
    num_random: int = 2,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Constructs a symmetric BigBird sparse attention pattern (Zaheer et al., NeurIPS 2020)
    combining global tokens, sliding window connections, and bidirectional random links.
    """
    if num_global < 0 or num_random < 0:
        raise ValueError(f"num_global and num_random must be non-negative, got {num_global}, {num_random}")
    rng = np.random.default_rng(seed)
    mask = build_sliding_window_mask(seq_len, window_radius)

    # Global tokens attend to all and are attended by all
    g = min(num_global, seq_len)
    mask[:g, :] = 1.0
    mask[:, :g] = 1.0

    # Symmetric random links per token
    if num_random > 0:
        for i in range(seq_len):
            candidates = np.where(mask[i] == 0.0)[0]
            if len(candidates) > 0:
                k = min(len(candidates), num_random)
                chosen = rng.choice(candidates, size=k, replace=False)
                mask[i, chosen] = 1.0
                mask[chosen, i] = 1.0

    return mask


class SparseMultiheadAttention(Module):
    """
    Sparse Multi-Head Self-Attention using BigBird connectivity patterns.
    Preserves full autograd through Q/K/V/Out projections.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        window_radius: int = 4,
        num_global: int = 2,
        num_random: int = 2
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_radius = window_radius
        self.num_global = num_global
        self.num_random = num_random

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (B, S, D) or (S, D)
        Returns:
            out: Tensor of shape (B, S, D) or (S, D)
        """
        orig_shape = list(x.shape)
        single = False
        if len(orig_shape) == 2:
            x_3d = x.unsqueeze(0)
            single = True
        elif len(orig_shape) == 3:
            x_3d = x
        else:
            raise ValueError(f"SparseMultiheadAttention requires 2D or 3D input tensor, got shape {orig_shape}")

        B, S, D = x_3d.shape
        if D != self.embed_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embed_dim}, got {D}")
        if S == 0 or B == 0:
            return x
        scale = 1.0 / math.sqrt(self.head_dim)

        q = self.q_proj(x_3d).reshape([B, S, self.num_heads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(x_3d).reshape([B, S, self.num_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(x_3d).reshape([B, S, self.num_heads, self.head_dim]).transpose(1, 2)

        mask = build_bigbird_mask(S, self.num_global, self.window_radius, self.num_random)
        mask_penalty = ((1.0 - mask) * -1e9).astype(np.float32)
        mask_t = _from_numpy(mask_penalty.reshape(1, 1, S, S))

        scores = q.matmul(k.transpose(2, 3)) * scale + mask_t
        attn_weights = scores.softmax(-1)

        context = attn_weights.matmul(v).transpose(1, 2).reshape([B, S, D])
        out = self.out_proj(context)

        if single:
            return out.squeeze(0)
        return out


__all__ = [
    "build_sliding_window_mask",
    "build_bigbird_mask",
    "SparseMultiheadAttention",
]
