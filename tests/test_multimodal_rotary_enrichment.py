"""Comprehensive tests for RotaryEmbedding2D, RotaryEmbedding3D, and CrossAttentionWithKVCache."""

import numpy as np
import pytest

from oxideon import (
    CrossAttentionWithKVCache,
    RotaryEmbedding2D,
    RotaryEmbedding3D,
    Tensor,
    from_numpy,
)


def test_rotary_embedding_2d_properties_and_forward():
    rope2d = RotaryEmbedding2D(16)
    assert rope2d.dim == 16
    assert rope2d.dim_h == 8
    assert rope2d.dim_w == 8

    # Custom partitions
    rope2d_custom = RotaryEmbedding2D.with_partitions(4, 12)
    assert rope2d_custom.dim == 16
    assert rope2d_custom.dim_h == 4
    assert rope2d_custom.dim_w == 12

    # Grid 4x4 = 16 tokens, batch=2, heads=2, head_dim=16
    np.random.seed(42)
    x_np = np.random.randn(2, 16, 2, 16).astype(np.float32)
    x = from_numpy(x_np)

    out = rope2d.forward(x, 4, 4)
    assert out.shape == (2, 16, 2, 16)

    # Check norm conservation: ||R x|| == ||x||
    in_norms = np.linalg.norm(x_np, axis=-1)
    out_norms = np.linalg.norm(out.to_numpy(), axis=-1)
    np.testing.assert_allclose(out_norms, in_norms, rtol=1e-5, atol=1e-5)


def test_rotary_embedding_2d_autograd():
    rope2d = RotaryEmbedding2D(8)
    # Grid 3x3 = 9 tokens, batch=1, heads=1, head_dim=8
    x_np = np.random.randn(1, 9, 1, 8).astype(np.float32)
    x = from_numpy(x_np).requires_grad_(True)

    out = rope2d.forward(x, 3, 3)
    loss = (out * out).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (1, 9, 1, 8)
    assert not np.isnan(x.grad.to_numpy()).any()
    # For squared loss ||R x||^2 = ||x||^2, grad wrt x is exactly 2 * x!
    np.testing.assert_allclose(x.grad.to_numpy(), 2.0 * x_np, rtol=1e-4, atol=1e-4)


def test_rotary_embedding_3d_properties_and_autograd():
    rope3d = RotaryEmbedding3D(16)
    # T=2, H=2, W=2 = 8 tokens
    x_np = np.random.randn(2, 8, 2, 16).astype(np.float32)
    x = from_numpy(x_np).requires_grad_(True)

    out = rope3d.forward(x, 2, 2, 2)
    assert out.shape == (2, 8, 2, 16)

    # Norm conservation
    in_norms = np.linalg.norm(x_np, axis=-1)
    out_norms = np.linalg.norm(out.to_numpy(), axis=-1)
    np.testing.assert_allclose(out_norms, in_norms, rtol=1e-5, atol=1e-5)

    # Autograd: d(||Rx||^2)/dx == 2x
    loss = (out * out).sum()
    loss.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.to_numpy(), 2.0 * x_np, rtol=1e-4, atol=1e-4)


def test_cross_attention_basic_and_masking():
    np.random.seed(123)
    cross_attn = CrossAttentionWithKVCache(
        query_dim=32,
        context_dim=48,
        num_heads=4,
        head_dim=8,
    )

    # Query: B=2, S_q=5, D_q=32
    # Context: B=2, S_c=10, D_c=48
    q_np = np.random.randn(2, 5, 32).astype(np.float32)
    c_np = np.random.randn(2, 10, 48).astype(np.float32)
    q = from_numpy(q_np)
    c = from_numpy(c_np)

    # 1. Unmasked forward
    out = cross_attn(q, context=c)
    assert out.shape == (2, 5, 32)
    assert not np.isnan(out.to_numpy()).any()

    # 2. Length-adaptive boolean mask: only first 6 context tokens valid
    mask = np.zeros((2, 5, 10), dtype=bool)
    mask[:, :, :6] = True
    out_masked = cross_attn(q, context=c, mask=mask)
    assert out_masked.shape == (2, 5, 32)
    assert not np.isnan(out_masked.to_numpy()).any()


def test_cross_attention_kv_cache_and_compression():
    cross_attn = CrossAttentionWithKVCache(
        query_dim=16,
        context_dim=16,
        num_heads=2,
        head_dim=8,
        compression="stride",
        compression_stride=2,
        max_cache_len=8,
    )

    q = from_numpy(np.random.randn(1, 1, 16).astype(np.float32))
    # Context step 1: 4 tokens -> compressed by stride 2 to 2 tokens
    c1 = from_numpy(np.random.randn(1, 4, 16).astype(np.float32))

    out1 = cross_attn(q, context=c1, use_cache=True)
    assert out1.shape == (1, 1, 16)
    assert cross_attn._cached_k is not None
    assert cross_attn._cached_k.shape[2] == 2  # 4 // 2 = 2

    # Context step 2: 4 more tokens -> cache expands then compresses
    c2 = from_numpy(np.random.randn(1, 4, 16).astype(np.float32))
    out2 = cross_attn(q, context=c2, use_cache=True)
    assert out2.shape == (1, 1, 16)
    assert cross_attn._cached_k.shape[2] == 4

    # Step 3: Generation without new context (attending to cached context)
    out3 = cross_attn(q, use_cache=False)
    assert out3.shape == (1, 1, 16)

    # Test cache reset
    cross_attn.reset_cache()
    assert cross_attn._cached_k is None
    assert cross_attn._cached_v is None


def test_cross_attention_autograd_gradients():
    np.random.seed(999)
    cross_attn = CrossAttentionWithKVCache(
        query_dim=16,
        context_dim=16,
        num_heads=2,
        head_dim=8,
    )

    q = from_numpy(np.random.randn(2, 4, 16).astype(np.float32)).requires_grad_(True)
    c = from_numpy(np.random.randn(2, 6, 16).astype(np.float32)).requires_grad_(True)

    out = cross_attn(q, context=c)
    loss = (out * out).sum()
    loss.backward()

    # Verify query and weight gradients exist
    assert q.grad is not None
    assert q.grad.shape == (2, 4, 16)
    assert cross_attn.out_proj.weight.grad is not None
    assert not np.isnan(q.grad.to_numpy()).any()
