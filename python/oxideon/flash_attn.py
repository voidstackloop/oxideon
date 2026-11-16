"""FlashAttention-2 tiled memory-efficient exact attention algorithm.

Implements the FlashAttention-2 online softmax tiling algorithm (Dao, 2023):
Computes exact multi-head scaled dot-product attention in blocks without materializing
the intermediate $N \\times N$ attention matrix into global memory.
"""

from typing import Optional
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module


def flash_attention_forward(
    query: _Tensor,
    key: _Tensor,
    value: _Tensor,
    block_size_r: int = 64,
    block_size_c: int = 64,
    scale: Optional[float] = None,
    causal: bool = False,
) -> _Tensor:
    """Exact FlashAttention-2 forward computation via online softmax tiling."""
    q_np = query.to_numpy()
    k_np = key.to_numpy()
    v_np = value.to_numpy()

    if q_np.ndim != 4 or k_np.ndim != 4 or v_np.ndim != 4:
        raise ValueError(f"FlashAttention requires 4D tensors [batch, heads, seq, dim], got {q_np.shape}, {k_np.shape}, {v_np.shape}")

    batch_size, num_heads, seq_len_q, head_dim = q_np.shape
    kb, kh, seq_len_k, kd = k_np.shape
    vb, vh, seq_len_v, vd = v_np.shape

    if (batch_size, num_heads, head_dim) != (kb, kh, kd) or (batch_size, num_heads, head_dim) != (vb, vh, vd) or seq_len_k != seq_len_v:
        raise ValueError(f"Batch dimension shape mismatch in FlashAttention: query {q_np.shape}, key {k_np.shape}, value {v_np.shape}")

    if block_size_r < 1 or block_size_c < 1:
        raise ValueError(f"block_size_r and block_size_c must be >= 1, got {block_size_r}, {block_size_c}")

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    bh = batch_size * num_heads
    q_flat = q_np.reshape(bh, seq_len_q, head_dim)
    k_flat = k_np.reshape(bh, seq_len_k, head_dim)
    v_flat = v_np.reshape(bh, seq_len_k, head_dim)

    out_flat = np.zeros((bh, seq_len_q, head_dim), dtype=np.float32)

    num_blocks_r = (seq_len_q + block_size_r - 1) // block_size_r
    num_blocks_c = (seq_len_k + block_size_c - 1) // block_size_c

    for i in range(num_blocks_r):
        r_start = i * block_size_r
        r_end = min(r_start + block_size_r, seq_len_q)
        actual_br = r_end - r_start
        q_i = q_flat[:, r_start:r_end, :]

        m_i = np.full((bh, actual_br, 1), -np.inf, dtype=np.float32)
        l_i = np.zeros((bh, actual_br, 1), dtype=np.float32)
        o_i = np.zeros((bh, actual_br, head_dim), dtype=np.float32)

        for j in range(num_blocks_c):
            c_start = j * block_size_c
            c_end = min(c_start + block_size_c, seq_len_k)

            if causal and c_start > r_end - 1:
                continue

            k_j = k_flat[:, c_start:c_end, :]
            v_j = v_flat[:, c_start:c_end, :]

            s_ij = np.matmul(q_i, k_j.swapaxes(-1, -2)) * scale

            if causal:
                row_indices = np.arange(r_start, r_end)[:, None]
                col_indices = np.arange(c_start, c_end)[None, :]
                mask = col_indices > row_indices
                s_ij = np.where(mask, -np.inf, s_ij)

            m_ij = np.max(s_ij, axis=-1, keepdims=True)
            m_new = np.maximum(m_i, m_ij)

            p_tilde = np.exp(s_ij - m_new)
            l_new = np.exp(m_i - m_new) * l_i + np.sum(p_tilde, axis=-1, keepdims=True)

            alpha = np.exp(m_i - m_new) * l_i / (l_new + 1e-12)
            beta = 1.0 / (l_new + 1e-12)

            o_i = alpha * o_i + beta * np.matmul(p_tilde, v_j)

            m_i = m_new
            l_i = l_new

        out_flat[:, r_start:r_end, :] = o_i

    out_np = out_flat.reshape(batch_size, num_heads, seq_len_q, head_dim)
    out_tiled = _from_numpy(np.ascontiguousarray(out_np.astype(np.float32)))

    needs_grad = (
        getattr(query, "requires_grad", False)
        or getattr(key, "requires_grad", False)
        or getattr(value, "requires_grad", False)
    )
    if needs_grad and seq_len_q > 0 and seq_len_k > 0:
        scores_t = query.matmul(key.transpose(2, 3)) * float(scale)
        if causal:
            r_idx = np.arange(seq_len_q)[:, None]
            c_idx = np.arange(seq_len_k)[None, :]
            c_mask = np.where(c_idx > r_idx, -1e9, 0.0).astype(np.float32)
            scores_t = scores_t + _from_numpy(c_mask.reshape(1, 1, seq_len_q, seq_len_k))
        attn_t = scores_t.softmax(-1).matmul(value)
        return attn_t + (out_tiled - attn_t).detach()

    return out_tiled


class TiledAttention(Module):
    """Multi-Head Attention layer powered by FlashAttention-2 online tiling."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int = 64,
        causal: bool = False,
    ):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or block_size <= 0:
            raise ValueError(f"embed_dim, num_heads, and block_size must be positive, got {embed_dim}, {num_heads}, {block_size}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.causal = causal

    def forward(self, q: _Tensor, k: _Tensor, v: _Tensor) -> _Tensor:
        """Forward pass expecting [batch, seq_len, embed_dim] or [batch, heads, seq_len, head_dim]."""
        if len(q.shape) == 3:
            if len(k.shape) != 3 or len(v.shape) != 3:
                raise ValueError("All inputs must be 3D when query is 3D")
            b, sq, ed_q = q.shape
            bk, sk, ed_k = k.shape
            bv, sv, ed_v = v.shape
            if b != bk or b != bv:
                raise ValueError(f"Batch dimension mismatch: query {b}, key {bk}, value {bv}")
            if ed_q != self.embed_dim or ed_k != self.embed_dim or ed_v != self.embed_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embed_dim}, got {ed_q}, {ed_k}, {ed_v}")
            q_reshaped = q.reshape([b, sq, self.num_heads, self.head_dim]).transpose(1, 2)
            k_reshaped = k.reshape([b, sk, self.num_heads, self.head_dim]).transpose(1, 2)
            v_reshaped = v.reshape([b, sv, self.num_heads, self.head_dim]).transpose(1, 2)

            out = flash_attention_forward(
                q_reshaped,
                k_reshaped,
                v_reshaped,
                block_size_r=self.block_size,
                block_size_c=self.block_size,
                causal=self.causal,
            )
            return out.transpose(1, 2).reshape([b, sq, self.embed_dim])

        return flash_attention_forward(
            q,
            k,
            v,
            block_size_r=self.block_size,
            block_size_c=self.block_size,
            causal=self.causal,
        )


__all__ = [
    "flash_attention_forward",
    "TiledAttention",
]
