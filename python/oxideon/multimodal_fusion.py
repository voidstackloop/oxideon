"""Multimodal fusion architectures for vision-language and audio-text integration.

Implements:
  - CrossAttentionFusion: Multi-head cross-attention fusion between distinct modalities.
  - GatedMultimodalFusion: Dynamic non-linear gating mechanism (Arevalo et al., 2017).
"""

from typing import Optional
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, LayerNorm, Parameter


class CrossAttentionFusion(Module):
    """Multi-Head Cross-Attention layer bridging query modality with key-value modality."""

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, query_tokens: _Tensor, kv_tokens: _Tensor) -> _Tensor:
        """Fuses query tokens [batch, seq_q, d_model] with context kv_tokens [batch, seq_kv, d_model]."""
        residual = query_tokens
        b, sq, _ = query_tokens.shape
        _, sk, _ = kv_tokens.shape
        if sk == 0:
            return self.norm(residual)
        if sq == 0:
            return residual

        q = self.q_proj(query_tokens).reshape([b, sq, self.num_heads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(kv_tokens).reshape([b, sk, self.num_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(kv_tokens).reshape([b, sk, self.num_heads, self.head_dim]).transpose(1, 2)

        scores = q.matmul(k.transpose(2, 3)) * self.scale
        attn = scores.softmax(-1)

        out = attn.matmul(v).transpose(1, 2).reshape([b, sq, self.d_model])
        out_proj = self.out_proj(out)
        return self.norm(residual + out_proj)


class GatedMultimodalFusion(Module):
    """Dynamic learnable gating fusion network between two feature modalities.

    $$h_a = \\tanh(W_a x_a), \\quad h_b = \\tanh(W_b x_b)$$
    $$z = \\sigma(W_z [x_a, x_b] + b_z)$$
    $$h_{fused} = z \\odot h_a + (1 - z) \\odot h_b$$
    """

    def __init__(self, dim_a: int, dim_b: int, out_dim: int):
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.out_dim = out_dim

        self.proj_a = Linear(dim_a, out_dim)
        self.proj_b = Linear(dim_b, out_dim)
        self.gate_linear = Linear(dim_a + dim_b, out_dim)

    def forward(self, x_a: _Tensor, x_b: _Tensor) -> _Tensor:
        """Fuses x_a [batch, dim_a] and x_b [batch, dim_b] into [batch, out_dim]."""
        is_1d = x_a.ndim == 1 and x_b.ndim == 1
        if is_1d:
            x_a = x_a.unsqueeze(0)
            x_b = x_b.unsqueeze(0)
        if x_a.shape[0] != x_b.shape[0]:
            raise ValueError(f"Batch dimension mismatch: x_a has batch {x_a.shape[0]}, x_b has batch {x_b.shape[0]}")

        h_a = self.proj_a(x_a).tanh()
        h_b = self.proj_b(x_b).tanh()

        concat = _Tensor.cat([x_a, x_b], dim=-1)
        gate = self.gate_linear(concat).clamp(-30.0, 30.0).sigmoid()

        fused = gate * h_a + (1.0 - gate) * h_b
        if is_1d:
            fused = fused.squeeze(0)
        return fused


__all__ = [
    "CrossAttentionFusion",
    "GatedMultimodalFusion",
]
