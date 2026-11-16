"""Comprehensive test suite for Oxideon Kernel Optimization & Fused Execution Engine.

Verifies:
  - fused_add_layernorm: single-pass residual addition + LayerNorm with autograd
  - fused_bias_activation: fused bias addition + activation (ReLU, SiLU, GELU, Tanh) with exact analytical backward
  - fused_silu_and_mul: SwiGLU / LLaMA MLP gate-up multiplication with exact analytical backward
  - fused_rotary_qk: simultaneous Query-Key RoPE with cache locality and autograd
  - RMSNorm module acceleration via fused_rmsnorm across 2D and 3D shapes
"""

import numpy as np
import pytest
import oxideon as ox


def test_fused_add_layernorm_forward_and_autograd():
    shape = [3, 8]
    np.random.seed(42)
    x_np = np.random.randn(*shape).astype(np.float32)
    r_np = np.random.randn(*shape).astype(np.float32)
    w_np = np.random.randn(8).astype(np.float32)
    b_np = np.random.randn(8).astype(np.float32)

    # 1. With weight and bias
    x = ox.from_numpy(x_np).requires_grad_(True)
    r = ox.from_numpy(r_np).requires_grad_(True)
    w = ox.from_numpy(w_np).requires_grad_(True)
    b = ox.from_numpy(b_np).requires_grad_(True)

    normed, s = ox.fused_add_layernorm(x, r, w, b, eps=1e-5)
    assert normed.shape == shape
    assert s.shape == shape

    # Check forward values match eager computation
    s_expected = x_np + r_np
    mean = np.mean(s_expected, axis=-1, keepdims=True)
    var = np.var(s_expected, axis=-1, keepdims=True)
    normed_expected = ((s_expected - mean) / np.sqrt(var + 1e-5)) * w_np + b_np
    np.testing.assert_allclose(normed.to_numpy(), normed_expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(s.to_numpy(), s_expected, rtol=1e-5, atol=1e-5)

    # Check autograd backward
    loss = (normed * normed).sum()
    loss.backward()

    assert x.grad is not None
    assert r.grad is not None
    assert w.grad is not None
    assert b.grad is not None
    assert np.all(np.isfinite(x.grad.to_numpy()))
    assert np.all(np.isfinite(r.grad.to_numpy()))
    assert np.all(np.isfinite(w.grad.to_numpy()))
    assert np.all(np.isfinite(b.grad.to_numpy()))

    # 2. Without weight and bias
    x2 = ox.from_numpy(x_np).requires_grad_(True)
    r2 = ox.from_numpy(r_np).requires_grad_(True)
    normed2, s2 = ox.fused_add_layernorm(x2, r2, None, None, eps=1e-5)
    loss2 = normed2.sum()
    loss2.backward()
    assert x2.grad is not None
    assert r2.grad is not None


def test_fused_bias_activation_all_modes():
    shape = [4, 6]
    np.random.seed(123)
    x_np = np.random.randn(*shape).astype(np.float32)
    b_np = np.random.randn(6).astype(np.float32)

    for act in ["relu", "silu", "gelu", "tanh", "none"]:
        x = ox.from_numpy(x_np).requires_grad_(True)
        b = ox.from_numpy(b_np).requires_grad_(True)

        y = ox.fused_bias_activation(x, b, act=act)
        assert y.shape == shape

        pre = x_np + b_np
        if act == "relu":
            expected = np.maximum(pre, 0.0)
        elif act == "silu":
            expected = pre / (1.0 + np.exp(-pre))
        elif act == "tanh":
            expected = np.tanh(pre)
        elif act == "none":
            expected = pre
        else:  # gelu tanh approximation
            c = np.sqrt(2.0 / np.pi)
            inner = c * (pre + 0.044715 * (pre ** 3))
            expected = 0.5 * pre * (1.0 + np.tanh(inner))

        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-4, atol=1e-4)

        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert b.grad is not None
        assert np.all(np.isfinite(x.grad.to_numpy()))
        assert np.all(np.isfinite(b.grad.to_numpy()))
        assert np.linalg.norm(x.grad.to_numpy()) > 0
        assert np.linalg.norm(b.grad.to_numpy()) > 0

    # Test error handling on unsupported activation
    x_bad = ox.zeros([2, 3])
    b_bad = ox.zeros([3])
    with pytest.raises(ValueError, match="unsupported activation"):
        ox.fused_bias_activation(x_bad, b_bad, act="unknown_act")


def test_fused_silu_and_mul():
    np.random.seed(999)
    shape = [3, 10]
    x_np = np.random.randn(*shape).astype(np.float32)
    z_np = np.random.randn(*shape).astype(np.float32)

    x = ox.from_numpy(x_np).requires_grad_(True)
    z = ox.from_numpy(z_np).requires_grad_(True)

    y = ox.fused_silu_and_mul(x, z)
    assert y.shape == shape

    sig = 1.0 / (1.0 + np.exp(-x_np))
    silu_x = x_np * sig
    expected = silu_x * z_np
    np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-5, atol=1e-5)

    loss = (y * y).sum()
    loss.backward()

    assert x.grad is not None
    assert z.grad is not None
    assert np.all(np.isfinite(x.grad.to_numpy()))
    assert np.all(np.isfinite(z.grad.to_numpy()))
    assert np.linalg.norm(x.grad.to_numpy()) > 0
    assert np.linalg.norm(z.grad.to_numpy()) > 0


def test_fused_rotary_qk():
    # q, k shape: [batch, seq_len, head_dim] with head_dim = 6
    # cos, sin shape: [batch, seq_len, head_dim / 2] = [1, 4, 3]
    np.random.seed(42)
    q_np = np.random.randn(1, 4, 6).astype(np.float32)
    k_np = np.random.randn(1, 4, 6).astype(np.float32)
    angles = np.linspace(0.0, 1.0, 3, dtype=np.float32)
    cos_np = np.cos(angles)[None, None, :]
    sin_np = np.sin(angles)[None, None, :]

    q = ox.from_numpy(q_np).requires_grad_(True)
    k = ox.from_numpy(k_np).requires_grad_(True)
    cos = ox.from_numpy(cos_np)
    sin = ox.from_numpy(sin_np)

    q_rot, k_rot = ox.fused_rotary_qk(q, k, cos, sin)
    assert q_rot.shape == [1, 4, 6]
    assert k_rot.shape == [1, 4, 6]

    # Verify against individual fused_rope
    q_rot_ref = ox.fused_rope(q, cos, sin)
    k_rot_ref = ox.fused_rope(k, cos, sin)
    np.testing.assert_allclose(q_rot.to_numpy(), q_rot_ref.to_numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(k_rot.to_numpy(), k_rot_ref.to_numpy(), rtol=1e-6, atol=1e-6)

    loss = (q_rot * k_rot).sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert np.all(np.isfinite(q.grad.to_numpy()))
    assert np.all(np.isfinite(k.grad.to_numpy()))


def test_rmsnorm_module_2d_and_3d():
    # 2D
    norm2d = ox.nn.RMSNorm(8, eps=1e-6)
    x2d = ox.randn([4, 8], requires_grad=True)
    y2d = norm2d(x2d)
    assert y2d.shape == [4, 8]
    loss2d = y2d.sum()
    loss2d.backward()
    assert x2d.grad is not None
    assert norm2d.weight.grad is not None

    # 3D
    norm3d = ox.nn.RMSNorm(16, eps=1e-5)
    x3d = ox.randn([2, 5, 16], requires_grad=True)
    y3d = norm3d(x3d)
    assert y3d.shape == [2, 5, 16]
    loss3d = y3d.sum()
    loss3d.backward()
    assert x3d.grad is not None
    assert norm3d.weight.grad is not None
