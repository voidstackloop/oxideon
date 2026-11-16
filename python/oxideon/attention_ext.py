"""Modern Attention Mechanisms and Interpretability Tools.

Implements:
11. AlibiPositionalBias: Attention with Linear Biases for zero-cost length extrapolation
12. SpatiotemporalRoPE: 3D Spatiotemporal RoPE for video and grid transformers
13. MultiQueryAttention (MQA): Single KV head shared across all query heads
14. SlidingWindowAttention: Local-global band attention with bounded receptive field
15. DifferentialAttention: Dual-softmax subtracted attention canceling attention noise
16. LogitLens: Intermediate layer representation probe for Transformer interpretability
17. MLAAttention: Multi-head Latent Attention with low-rank KV and decoupled RoPE
18. NativeSparseAttention: Compressed + top-k selected + sliding branches with gating
"""

from typing import Optional, Tuple, List, Union, Any
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, LayerNorm, Parameter, _check_repeats, _check_elements, _check_positive


class AlibiPositionalBias(Module):
    """Attention with Linear Biases (ALiBi) (Press et al., 2022)."""

    def __init__(self, num_heads: int):
        super().__init__()
        num_heads = _check_repeats("num_heads", num_heads)
        self.num_heads = num_heads
        i = np.arange(num_heads, dtype=np.float64)
        self.slopes = np.power(2.0, -8.0 * (i + 1) / num_heads).astype(np.float32)

    def forward(self, seq_len: int) -> _Tensor:
        """Returns ALiBi bias matrix [num_heads, seq_len, seq_len]."""
        seq_len = _check_elements("seq_len", seq_len)
        i = np.arange(seq_len)[:, None]
        j = np.arange(seq_len)[None, :]
        dist = (j - i).astype(np.float32)

        bias = self.slopes[:, None, None] * dist[None, :, :]
        return _from_numpy(np.ascontiguousarray(bias))


class SpatiotemporalRoPE(Module):
    """3D Spatiotemporal Rotary Position Embedding for Video and Spatial Volumes."""

    def __init__(self, dim: int):
        super().__init__()
        dim = _check_elements("dim", dim)
        self.dim = dim
        self.dim_per_axis = dim // 3
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, self.dim_per_axis, 2, dtype=np.float32) / self.dim_per_axis))
        self.inv_freq = inv_freq

    def forward(self, t: int, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates 3D cos and sin rotation matrices for coordinates (t, h, w)."""
        t = _check_elements("t", t)
        h = _check_elements("h", h)
        w = _check_elements("w", w)
        grid_t = np.arange(t, dtype=np.float32)[:, None] * self.inv_freq[None, :]
        grid_h = np.arange(h, dtype=np.float32)[:, None] * self.inv_freq[None, :]
        grid_w = np.arange(w, dtype=np.float32)[:, None] * self.inv_freq[None, :]

        sin_t, cos_t = np.sin(grid_t), np.cos(grid_t)
        sin_h, cos_h = np.sin(grid_h), np.cos(grid_h)
        sin_w, cos_w = np.sin(grid_w), np.cos(grid_w)

        return (cos_t, cos_h, cos_w), (sin_t, sin_h, sin_w)


class MultiQueryAttention(Module):
    """Multi-Query Attention (MQA) with single Key/Value head shared across all Query heads."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = Linear(hidden_dim, hidden_dim)
        self.k_proj = Linear(hidden_dim, self.head_dim)
        self.v_proj = Linear(hidden_dim, self.head_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)

    def forward(self, x: _Tensor) -> _Tensor:
        """Args: x: [batch, seq_len, hidden_dim] -> [batch, seq_len, hidden_dim]"""
        b, s, d = x.shape
        if b == 0 or s == 0:
            return x
        q = self.q_proj(x).reshape([b, s, self.num_heads, self.head_dim]).transpose(1, 2)  # [b, H, s, Dh]
        k = self.k_proj(x).unsqueeze(1)  # [b, 1, s, Dh]
        v = self.v_proj(x).unsqueeze(1)  # [b, 1, s, Dh]

        k_exp = _Tensor.cat([k for _ in range(self.num_heads)], dim=1)
        v_exp = _Tensor.cat([v for _ in range(self.num_heads)], dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = q.matmul(k_exp.transpose(2, 3)) * scale
        attn = scores.softmax(-1)

        out = attn.matmul(v_exp).transpose(1, 2).reshape([b, s, d])
        return self.out_proj(out)


class SlidingWindowAttention(Module):
    """Sliding Window Local Attention with bounded receptive field W."""

    def __init__(self, hidden_dim: int, num_heads: int, window_size: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size

        self.q_proj = Linear(hidden_dim, hidden_dim)
        self.k_proj = Linear(hidden_dim, hidden_dim)
        self.v_proj = Linear(hidden_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)

    def forward(self, x: _Tensor) -> _Tensor:
        b, s, d = x.shape
        if b == 0 or s == 0:
            return x
        q = self.q_proj(x).reshape([b, s, self.num_heads, self.head_dim]).transpose(1, 2)
        k = self.k_proj(x).reshape([b, s, self.num_heads, self.head_dim]).transpose(1, 2)
        v = self.v_proj(x).reshape([b, s, self.num_heads, self.head_dim]).transpose(1, 2)

        scores = q.matmul(k.transpose(2, 3)) * (1.0 / math.sqrt(self.head_dim))

        i = np.arange(s)[:, None]
        j = np.arange(s)[None, :]
        mask = (j <= i) & (j >= i - self.window_size)
        penalty = np.where(mask, 0.0, -1e9).astype(np.float32)
        scores = scores + _from_numpy(penalty.reshape(1, 1, s, s))

        attn = scores.softmax(-1)
        out = attn.matmul(v).transpose(1, 2).reshape([b, s, d])
        return self.out_proj(out)


class DifferentialAttention(Module):
    """Differential Attention canceling noise via dual-softmax subtraction (Ye et al., 2024)."""

    def __init__(self, hidden_dim: int, num_heads: int, lambda_init: float = 0.8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q1 = Linear(hidden_dim, hidden_dim)
        self.k1 = Linear(hidden_dim, hidden_dim)
        self.q2 = Linear(hidden_dim, hidden_dim)
        self.k2 = Linear(hidden_dim, hidden_dim)
        self.v = Linear(hidden_dim, hidden_dim)
        self.out = Linear(hidden_dim, hidden_dim)

        self.lambda_param = Parameter(_from_numpy(np.array([lambda_init], dtype=np.float32)))

    def forward(self, x: _Tensor) -> _Tensor:
        b, s, d = x.shape
        if b == 0 or s == 0:
            return x
        q1 = self.q1(x).reshape([b, s, self.num_heads, self.head_dim]).transpose(1, 2)
        k1 = self.k1(x).reshape([b, s, self.num_heads, self.head_dim]).transpose(1, 2)
        q2 = self.q2(x).reshape([b, s, self.num_heads, self.head_dim]).transpose(1, 2)
        k2 = self.k2(x).reshape([b, s, self.num_heads, self.head_dim]).transpose(1, 2)
        v = self.v(x).reshape([b, s, self.num_heads, self.head_dim]).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        s1 = q1.matmul(k1.transpose(2, 3)) * scale
        s2 = q2.matmul(k2.transpose(2, 3)) * scale

        a1 = s1.softmax(-1)
        a2 = s2.softmax(-1)

        lam = self.lambda_param.reshape([1, 1, 1, 1])
        diff_attn = a1 - a2 * lam

        out = diff_attn.matmul(v).transpose(1, 2).reshape([b, s, d])
        return self.out(out)


class LogitLens:
    """Probes intermediate transformer representations by projecting hidden states to the vocabulary."""

    @staticmethod
    def inspect_layer(
        hidden_state: _Tensor,
        lm_head_weight: _Tensor,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Projects hidden states [batch, seq_len, hidden_dim] onto vocab logits [batch, seq_len, vocab_size]."""
        h = hidden_state.to_numpy()
        w = lm_head_weight.to_numpy()
        logits = h @ w.T
        top_indices = np.argsort(-logits, axis=-1)[..., :top_k]
        rows = np.take_along_axis(logits, top_indices, axis=-1)
        return top_indices, rows


def _apply_rope_np(t: np.ndarray) -> np.ndarray:
    """NeoX-style rotary embedding over the last dim (even), positions 0..s."""
    b, h, s, r = t.shape
    pos = np.arange(s, dtype=np.float64)[:, None]
    freq = 1.0 / (10000.0 ** (np.arange(0, r, 2, dtype=np.float64) / r))
    ang = pos * freq
    cos = np.cos(ang)[None, None, :, :]
    sin = np.sin(ang)[None, None, :, :]
    t0 = t[..., 0::2]
    t1 = t[..., 1::2]
    out = np.empty_like(t)
    out[..., 0::2] = t0 * cos - t1 * sin
    out[..., 1::2] = t0 * sin + t1 * cos
    return out


class MLAAttention(Module):
    """Multi-head Latent Attention (DeepSeek-V2 style): low-rank KV + decoupled RoPE."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        q_lora_rank: int = 32,
        kv_lora_rank: int = 32,
        qk_rope_head_dim: int = 16,
        qk_nope_head_dim: int = 32,
        v_head_dim: int = 32,
    ):
        super().__init__()
        self.hidden_dim = _check_positive("hidden_dim", hidden_dim)
        self.num_heads = _check_positive("num_heads", num_heads)
        self.q_lora_rank = _check_positive("q_lora_rank", q_lora_rank)
        self.kv_lora_rank = _check_positive("kv_lora_rank", kv_lora_rank)
        self.qk_rope_head_dim = _check_positive("qk_rope_head_dim", qk_rope_head_dim)
        self.qk_nope_head_dim = _check_positive("qk_nope_head_dim", qk_nope_head_dim)
        self.v_head_dim = _check_positive("v_head_dim", v_head_dim)
        if self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even for rotary pairs")
        if self.num_heads > 512:
            raise ValueError(f"num_heads ({num_heads}) exceeds maximum supported 512")
        q_up_dim = num_heads * (qk_nope_head_dim + qk_rope_head_dim)
        if q_up_dim > 262144:
            raise ValueError(f"total projected Q dimension ({q_up_dim}) exceeds maximum 262144")
        if num_heads * v_head_dim > 262144:
            raise ValueError(f"total projected V dimension ({num_heads * v_head_dim}) exceeds maximum 262144")

        self.w_q_down = Linear(hidden_dim, q_lora_rank)
        self.q_norm = LayerNorm(q_lora_rank)
        self.w_q_up = Linear(q_lora_rank, num_heads * (qk_nope_head_dim + qk_rope_head_dim))
        self.w_kv_down = Linear(hidden_dim, kv_lora_rank)
        self.kv_norm = LayerNorm(kv_lora_rank)
        self.w_k_nope = Linear(kv_lora_rank, num_heads * qk_nope_head_dim)
        self.w_v = Linear(kv_lora_rank, num_heads * v_head_dim)
        self.w_k_rope = Linear(hidden_dim, qk_rope_head_dim)
        self.w_o = Linear(num_heads * v_head_dim, hidden_dim)

    def forward(self, x: _Tensor, causal: bool = True) -> _Tensor:
        """Args: x: [batch, seq_len, hidden_dim] -> [batch, seq_len, hidden_dim]."""
        b, s, d = x.shape
        if d != self.hidden_dim:
            raise ValueError(f"hidden dim {d} != configured {self.hidden_dim}")
        h, nope, rope, v = self.num_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
        c_q = self.q_norm(self.w_q_down(x)).to_numpy()
        q = self.w_q_up(_from_numpy(np.ascontiguousarray(c_q))).to_numpy()
        q = q.reshape(b, s, h, nope + rope).transpose(0, 2, 1, 3)
        c_kv = self.kv_norm(self.w_kv_down(x)).to_numpy()
        k_nope = self.w_k_nope(_from_numpy(np.ascontiguousarray(c_kv))).to_numpy()
        k_nope = k_nope.reshape(b, s, h, nope).transpose(0, 2, 1, 3)
        vv = self.w_v(_from_numpy(np.ascontiguousarray(c_kv))).to_numpy()
        vv = vv.reshape(b, s, h, v).transpose(0, 2, 1, 3)
        k_rope = self.w_k_rope(x).to_numpy()[:, None, :, :]

        q_rope = _apply_rope_np(np.ascontiguousarray(q[..., nope:]))
        k_rope = _apply_rope_np(np.ascontiguousarray(k_rope))
        k_rope = np.broadcast_to(k_rope, q_rope.shape)
        q_nope = q[..., :nope]

        scale = 1.0 / math.sqrt(nope + rope)
        scores = (np.einsum("bhid,bhjd->bhij", q_nope, k_nope) + np.einsum("bhid,bhjd->bhij", q_rope, k_rope)) * scale
        if causal:
            i = np.arange(s)[:, None]
            j = np.arange(s)[None, :]
            scores = np.where((j <= i)[None, None, :, :], scores, -1e9)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-9)

        out = np.einsum("bhij,bhjd->bhid", attn, vv).transpose(0, 2, 1, 3).reshape(b, s, h * v)
        return self.w_o(_from_numpy(np.ascontiguousarray(out)))


class NativeSparseAttention(Module):
    """Native Sparse Attention lite: compressed + top-k selected + sliding branches."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        block_size: int = 32,
        top_blocks: int = 2,
        window_size: int = 64,
    ):
        super().__init__()
        self.hidden_dim = _check_positive("hidden_dim", hidden_dim)
        self.num_heads = _check_positive("num_heads", num_heads)
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim {hidden_dim} must split over {num_heads} heads")
        self.head_dim = hidden_dim // num_heads
        self.block_size = _check_positive("block_size", block_size)
        self.top_blocks = _check_positive("top_blocks", top_blocks)
        self.window_size = _check_positive("window_size", window_size)

        self.q_proj = Linear(hidden_dim, hidden_dim)
        self.k_proj = Linear(hidden_dim, hidden_dim)
        self.v_proj = Linear(hidden_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)
        self.gate = Linear(hidden_dim, 3)

    def forward(self, x: _Tensor) -> _Tensor:
        """Args: x: [batch, seq_len, hidden_dim] -> [batch, seq_len, hidden_dim]."""
        b, s, d = x.shape
        if d != self.hidden_dim:
            raise ValueError(f"hidden dim {d} != configured {self.hidden_dim}")
        h, dh = self.num_heads, self.head_dim
        bs = self.block_size
        q = self.q_proj(x).to_numpy().reshape(b, s, h, dh).transpose(0, 2, 1, 3)
        k = self.k_proj(x).to_numpy().reshape(b, s, h, dh).transpose(0, 2, 1, 3)
        v = self.v_proj(x).to_numpy().reshape(b, s, h, dh).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(dh)
        pos_i = np.arange(s)[:, None]
        pos_j = np.arange(s)[None, :]

        nb = (s + bs - 1) // bs
        pad = nb * bs - s
        if pad:
            k = np.concatenate([k, np.zeros((b, h, pad, dh), dtype=k.dtype)], axis=2)
            v = np.concatenate([v, np.zeros((b, h, pad, dh), dtype=v.dtype)], axis=2)
        kc = k.reshape(b, h, nb, bs, dh).mean(axis=3)
        vc = v.reshape(b, h, nb, bs, dh).mean(axis=3)
        scores_c = np.einsum("bhid,bhjd->bhij", q, kc) * scale
        block_end = (np.arange(nb) + 1) * bs - 1
        mask_c = block_end[None, :] <= pos_i
        scores_c = np.where(mask_c[None, None, :, :], scores_c, -1e9)
        exp_c = np.exp(scores_c - np.max(scores_c, axis=-1, keepdims=True))
        attn_c = exp_c / (np.sum(exp_c, axis=-1, keepdims=True) + 1e-9)
        out_c = np.einsum("bhij,bhjd->bhid", attn_c, vc)
        out_c = out_c * mask_c.any(axis=-1)[None, None, :, None]

        take = min(self.top_blocks, nb)
        top_idx = np.argsort(-scores_c, axis=-1, kind="stable")[..., :take]
        block_of_j = np.minimum(pos_j // bs, nb - 1)
        sel = (block_of_j[None, None, None, :] == top_idx[..., None]).any(axis=-2)
        sel = sel & (pos_j <= pos_i)
        scores_s = np.einsum("bhid,bhjd->bhij", q, k[:, :, :s, :]) * scale
        scores_s = np.where(sel, scores_s, -1e9)
        exp_s = np.exp(scores_s - np.max(scores_s, axis=-1, keepdims=True))
        attn_s = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-9)
        out_s = np.einsum("bhij,bhjd->bhid", attn_s, v[:, :, :s, :])
        out_s = out_s * sel.any(axis=-1)[..., None]

        mask_w = (pos_j <= pos_i) & (pos_j > pos_i - self.window_size)
        scores_w = np.einsum("bhid,bhjd->bhij", q, k[:, :, :s, :]) * scale
        scores_w = np.where(mask_w[None, None, :, :], scores_w, -1e9)
        exp_w = np.exp(scores_w - np.max(scores_w, axis=-1, keepdims=True))
        attn_w = exp_w / (np.sum(exp_w, axis=-1, keepdims=True) + 1e-9)
        out_w = np.einsum("bhij,bhjd->bhid", attn_w, v[:, :, :s, :])

        g = self.gate(x).to_numpy()
        g = np.exp(g - np.max(g, axis=-1, keepdims=True))
        g = g / (np.sum(g, axis=-1, keepdims=True) + 1e-9)
        mixed = (
            g[:, :, 0][:, None, :, None] * out_c
            + g[:, :, 1][:, None, :, None] * out_s
            + g[:, :, 2][:, None, :, None] * out_w
        )
        out = mixed.transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.out_proj(_from_numpy(np.ascontiguousarray(out)))


class CrossAttentionWithKVCache(Module):
    """Multi-Head Cross-Attention with Key-Value Cache compression and length-adaptive masking.

    Allows queries from target modality (e.g. decoder or generation state) to attend to
    encoder / context representations (e.g. prompt, audio, vision embeddings).

    Features:
    - Incremental KV Cache for multi-step autoregressive generation or diffusion steps.
    - KV Cache Compression:
        - "none": Full KV caching.
        - "stride" / "pooling": Downsamples context keys/values along sequence dimension.
        - "max_len": Rolling FIFO buffer evicting oldest tokens past `max_cache_len`.
    - Length-adaptive masks supporting variable context lengths and custom attention masks.
    - Fully end-to-end differentiable autograd backpropagation through queries, keys, and values.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        compression: str = "none",
        compression_stride: int = 2,
        max_cache_len: Optional[int] = None,
    ):
        super().__init__()
        self.query_dim = _check_positive("query_dim", query_dim)
        self.context_dim = query_dim if context_dim is None else _check_positive("context_dim", context_dim)
        self.num_heads = _check_positive("num_heads", num_heads)
        self.head_dim = (query_dim // num_heads) if head_dim is None else _check_positive("head_dim", head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = float(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.compression = compression.lower()
        self.compression_stride = max(1, int(compression_stride))
        self.max_cache_len = max_cache_len

        self.q_proj = Linear(self.query_dim, self.inner_dim)
        self.k_proj = Linear(self.context_dim, self.inner_dim)
        self.v_proj = Linear(self.context_dim, self.inner_dim)
        self.out_proj = Linear(self.inner_dim, self.query_dim)

        self._cached_k: Optional[_Tensor] = None
        self._cached_v: Optional[_Tensor] = None

    def reset_cache(self) -> None:
        """Clears stored Key-Value cache."""
        self._cached_k = None
        self._cached_v = None

    def _compress_kv(self, k: _Tensor, v: _Tensor) -> Tuple[_Tensor, _Tensor]:
        """Applies compression policy to Key and Value tensors."""
        if self.compression in ("stride", "pooling") and self.compression_stride > 1:
            # k, v: [B, H, S, D]
            s = k.shape[2]
            stride = self.compression_stride
            if s >= stride:
                trim_s = (s // stride) * stride
                k_trim = k.narrow(2, 0, trim_s)
                v_trim = v.narrow(2, 0, trim_s)
                b, h, _, d = k.shape
                k_comp = k_trim.reshape([b, h, trim_s // stride, stride, d]).mean(dim=3)
                v_comp = v_trim.reshape([b, h, trim_s // stride, stride, d]).mean(dim=3)
                if trim_s < s:
                    rem = s - trim_s
                    k_rem = k.narrow(2, trim_s, rem).mean(dim=2).unsqueeze(2)
                    v_rem = v.narrow(2, trim_s, rem).mean(dim=2).unsqueeze(2)
                    k = _Tensor.cat([k_comp, k_rem], dim=2)
                    v = _Tensor.cat([v_comp, v_rem], dim=2)
                else:
                    k = k_comp
                    v = v_comp

        if self.max_cache_len is not None and self.max_cache_len > 0:
            if k.shape[2] > self.max_cache_len:
                start = k.shape[2] - self.max_cache_len
                k = k.narrow(2, start, self.max_cache_len)
                v = v.narrow(2, start, self.max_cache_len)

        return k, v

    def forward(
        self,
        x: _Tensor,
        context: Optional[_Tensor] = None,
        mask: Optional[Union[_Tensor, np.ndarray]] = None,
        use_cache: bool = False,
    ) -> _Tensor:
        """Performs Cross-Attention.

        Args:
            x: Query input [batch, q_len, query_dim]
            context: Optional context input [batch, kv_len, context_dim]
            mask: Optional attention mask [batch, q_len, kv_len] or broadcastable
            use_cache: If True, stores or updates key/value cache
        """
        b, s_q, _ = x.shape
        h, dh = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape([b, s_q, h, dh]).transpose(1, 2)

        if context is not None:
            b_c, s_c, _ = context.shape
            k_cur = self.k_proj(context).reshape([b_c, s_c, h, dh]).transpose(1, 2)
            v_cur = self.v_proj(context).reshape([b_c, s_c, h, dh]).transpose(1, 2)
            k_cur, v_cur = self._compress_kv(k_cur, v_cur)

            if use_cache:
                if self._cached_k is not None:
                    k = _Tensor.cat([self._cached_k, k_cur], dim=2)
                    v = _Tensor.cat([self._cached_v, v_cur], dim=2)
                    if self.max_cache_len is not None and k.shape[2] > self.max_cache_len:
                        start = k.shape[2] - self.max_cache_len
                        k = k.narrow(2, start, self.max_cache_len)
                        v = v.narrow(2, start, self.max_cache_len)
                else:
                    k, v = k_cur, v_cur
                self._cached_k = k
                self._cached_v = v
            else:
                k, v = k_cur, v_cur
        elif self._cached_k is not None:
            k = self._cached_k
            v = self._cached_v
        else:
            # Self-attention fallback
            k = self.k_proj(x).reshape([b, s_q, h, dh]).transpose(1, 2)
            v = self.v_proj(x).reshape([b, s_q, h, dh]).transpose(1, 2)
            k, v = self._compress_kv(k, v)
            if use_cache:
                self._cached_k = k
                self._cached_v = v

        scores = q.matmul(k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_arr = mask.to_numpy() if isinstance(mask, _Tensor) else np.asarray(mask)
            if mask_arr.ndim == 2:
                # [s_q, s_k] -> [1, 1, s_q, s_k]
                mask_arr = mask_arr[None, None, :, :]
            elif mask_arr.ndim == 3:
                # [b, s_q, s_k] -> [b, 1, s_q, s_k]
                mask_arr = mask_arr[:, None, :, :]

            if mask_arr.dtype == bool or np.issubdtype(mask_arr.dtype, np.bool_):
                keep = _from_numpy(np.ascontiguousarray(mask_arr, dtype=np.float32))
                scores = scores * keep + (1.0 - keep) * -1e9
            elif (mask_arr == 0).any() and not (mask_arr < 0).any():
                keep = _from_numpy(np.ascontiguousarray(mask_arr > 0.5, dtype=np.float32))
                scores = scores * keep + (1.0 - keep) * -1e9
            else:
                bias = _from_numpy(np.ascontiguousarray(mask_arr, dtype=np.float32))
                scores = scores + bias

        weights = scores.softmax(-1)
        attn_out = weights.matmul(v).transpose(1, 2).reshape([b, s_q, self.inner_dim])
        return self.out_proj(attn_out)


__all__ = [
    "AlibiPositionalBias",
    "SpatiotemporalRoPE",
    "MultiQueryAttention",
    "SlidingWindowAttention",
    "DifferentialAttention",
    "LogitLens",
    "MLAAttention",
    "NativeSparseAttention",
    "CrossAttentionWithKVCache",
]
