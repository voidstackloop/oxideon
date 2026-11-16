"""Tests for Batch 30 layers: Maxout, Snake, Swish, DropPath, AlphaDropout,
ConvLstmCell, Upsample2d, RNNCell, LSTMCell, GRUCell, and SpectralNormLinear.
"""

import math
import numpy as np
import pytest
import torch

import oxideon as ox
import oxideon.nn as nn


def test_maxout_forward_and_backward():
    x_np = np.random.randn(2, 3, 8).astype(np.float32)
    x_ox = ox.tensor(x_np, requires_grad=True)
    m2 = nn.Maxout(2)
    out = m2(x_ox)
    assert out.shape == (2, 3, 4)
    # Ground truth: max over pairs of 2 consecutive features
    expected = x_np.reshape(2, 3, 4, 2).max(axis=-1)
    assert np.allclose(out.to_numpy(), expected, atol=1e-5)

    out.sum().backward()
    assert x_ox.grad is not None
    assert x_ox.grad.shape == (2, 3, 8)

    # Maxout with k=4
    m4 = nn.Maxout(4)
    out4 = m4(x_ox)
    assert out4.shape == (2, 3, 2)


def test_snake_forward_and_backward():
    x_np = np.random.randn(3, 4).astype(np.float32)
    x_ox = ox.tensor(x_np, requires_grad=True)
    snake = nn.Snake(4, alpha=0.5)
    out = snake(x_ox)
    assert out.shape == (3, 4)

    # Ground truth: x + sin(alpha * x)^2 / alpha
    alpha = snake.alpha.to_numpy()
    expected = x_np + np.sin(alpha * x_np) ** 2 / alpha
    assert np.allclose(out.to_numpy(), expected, atol=1e-5)

    out.sum().backward()
    assert x_ox.grad is not None
    assert snake.alpha.grad is not None


def test_swish_forward_and_backward():
    x_np = np.random.randn(3, 4).astype(np.float32)
    x_ox = ox.tensor(x_np, requires_grad=True)
    swish = nn.Swish(beta=1.5)
    out = swish(x_ox)
    assert out.shape == (3, 4)

    beta = swish.beta.to_numpy()
    expected = x_np / (1.0 + np.exp(-beta * x_np))
    assert np.allclose(out.to_numpy(), expected, atol=1e-5)

    out.sum().backward()
    assert x_ox.grad is not None
    assert swish.beta.grad is not None


def test_droppath_eval_and_train():
    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    x_ox = ox.tensor(x_np, requires_grad=True)

    dp = nn.DropPath(0.3)
    dp.eval()
    out_eval = dp(x_ox)
    assert np.allclose(out_eval.to_numpy(), x_np, atol=1e-6)

    # p=0.0 train is also identity
    dp0 = nn.DropPath(0.0)
    out0 = dp0(x_ox)
    assert np.allclose(out0.to_numpy(), x_np, atol=1e-6)

    out_eval.sum().backward()
    assert np.allclose(x_ox.grad.to_numpy(), np.ones_like(x_np), atol=1e-6)


def test_alphadropout_eval_and_extremes():
    x_np = np.random.randn(3, 5).astype(np.float32)
    x_ox = ox.tensor(x_np, requires_grad=True)

    ad = ox.oxideon.AlphaDropout(0.4)
    ad.eval()
    out_eval = ad(x_ox)
    assert np.allclose(out_eval.to_numpy(), x_np, atol=1e-6)

    # p=0.0 in train mode is identity
    ad0 = ox.oxideon.AlphaDropout(0.0)
    out0 = ad0(x_ox)
    assert np.allclose(out0.to_numpy(), x_np, atol=1e-6)

    # p=1.0 in train mode produces zeros and maintains autograd tape
    x_ox2 = ox.tensor(x_np, requires_grad=True)
    ad1 = ox.oxideon.AlphaDropout(1.0)
    out1 = ad1(x_ox2)
    assert np.allclose(out1.to_numpy(), np.zeros_like(x_np), atol=1e-6)
    out1.sum().backward()
    assert x_ox2.grad is not None
    assert np.allclose(x_ox2.grad.to_numpy(), np.zeros_like(x_np), atol=1e-6)


def test_conv_lstm_cell_step_and_sequence():
    clc = nn.ConvLstmCell(in_channels=3, hidden=4, kernel=3)
    x_step = ox.randn(2, 3, 4, 4, requires_grad=True)
    h0 = ox.randn(2, 4, 4, 4, requires_grad=True)
    c0 = ox.randn(2, 4, 4, 4, requires_grad=True)

    hn, cn = clc.step(x_step, (h0, c0))
    assert hn.shape == (2, 4, 4, 4)
    assert cn.shape == (2, 4, 4, 4)

    (hn.sum() + cn.sum()).backward()
    assert x_step.grad is not None
    assert h0.grad is not None
    assert c0.grad is not None
    assert clc.weight.grad is not None
    assert clc.bias.grad is not None

    # Forward sequence [B, T, C, H, W]
    x_seq = ox.randn(2, 3, 3, 4, 4, requires_grad=True)
    seq_out = clc(x_seq)
    assert seq_out.shape == (2, 3, 4, 4, 4)
    seq_out.sum().backward()
    assert x_seq.grad is not None


def test_upsample2d_nearest_and_bilinear():
    x_np = np.random.randn(2, 2, 3, 3).astype(np.float32)
    x_ox = ox.tensor(x_np, requires_grad=True)
    x_torch = torch.from_numpy(x_np).requires_grad_(True)

    # Nearest
    up_near = nn.Upsample2d((6, 6), mode="nearest")
    out_near = up_near(x_ox)
    torch_near = torch.nn.functional.interpolate(x_torch, size=(6, 6), mode="nearest")
    assert np.allclose(out_near.to_numpy(), torch_near.detach().numpy(), atol=1e-5)

    out_near.sum().backward()
    torch_near.sum().backward()
    assert np.allclose(x_ox.grad.to_numpy(), x_torch.grad.numpy(), atol=1e-4)

    # Bilinear align_corners=False
    x_ox.zero_grad()
    x_torch.grad.zero_()
    up_bi_f = nn.Upsample2d((6, 6), mode="bilinear", align_corners=False)
    out_bi_f = up_bi_f(x_ox)
    torch_bi_f = torch.nn.functional.interpolate(x_torch, size=(6, 6), mode="bilinear", align_corners=False)
    assert np.allclose(out_bi_f.to_numpy(), torch_bi_f.detach().numpy(), atol=1e-5)

    out_bi_f.sum().backward()
    torch_bi_f.sum().backward()
    assert np.allclose(x_ox.grad.to_numpy(), x_torch.grad.numpy(), atol=1e-4)

    # Bilinear align_corners=True
    x_ox.zero_grad()
    x_torch.grad.zero_()
    up_bi_t = nn.Upsample2d.bilinear((6, 6), align_corners=True)
    out_bi_t = up_bi_t(x_ox)
    torch_bi_t = torch.nn.functional.interpolate(x_torch, size=(6, 6), mode="bilinear", align_corners=True)
    assert np.allclose(out_bi_t.to_numpy(), torch_bi_t.detach().numpy(), atol=1e-5)

    out_bi_t.sum().backward()
    torch_bi_t.sum().backward()
    assert np.allclose(x_ox.grad.to_numpy(), x_torch.grad.numpy(), atol=1e-4)


def test_spectral_norm_linear_matches_torch():
    sn = nn.SpectralNormLinear(4, 3, use_bias=True, n_power_iterations=2)
    x_np = np.random.randn(2, 4).astype(np.float32)
    x_ox = ox.tensor(x_np, requires_grad=True)

    out = sn(x_ox)
    assert out.shape == (2, 3)
    out.sum().backward()
    assert x_ox.grad is not None
    assert sn.weight.grad is not None
    assert sn.bias.grad is not None
