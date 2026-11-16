"""Comprehensive tests for Oxideon Functional API (ox.nn.functional / ox.F).

Validates scaled dot-product attention, 1D/2D/3D convolutions & transposed convolutions,
pooling variants, normalization algorithms, vision transformations, embeddings,
advanced activations, and loss functions.
"""

import math
import numpy as np
import pytest

import oxideon as ox
import oxideon.nn.functional as F


def test_scaled_dot_product_attention_causal_and_scaling():
    B, H, L, D = 2, 4, 8, 16
    q = ox.randn(B, H, L, D)
    k = ox.randn(B, H, L, D)
    v = ox.randn(B, H, L, D)

    # Standard SDPA
    out = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, L, D)

    # Causal SDPA
    causal_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert causal_out.shape == (B, H, L, D)

    # Custom scale
    scaled_out = F.scaled_dot_product_attention(q, k, v, scale=0.5)
    assert scaled_out.shape == (B, H, L, D)


def test_convolutions_1d_2d_3d_and_transpose():
    # Conv1D: (N, C, L)
    x1 = ox.randn(2, 3, 16)
    w1 = ox.randn(4, 3, 3)
    b1 = ox.zeros(4)
    y1 = F.conv1d(x1, w1, b1, stride=1, padding=1)
    assert y1.shape == (2, 4, 16)

    # Conv2D: (N, C, H, W)
    x2 = ox.randn(2, 3, 16, 16)
    w2 = ox.randn(8, 3, 3, 3)
    y2 = F.conv2d(x2, w2, padding=1)
    assert y2.shape == (2, 8, 16, 16)

    # Conv3D: (N, C, D, H, W)
    x3 = ox.randn(1, 2, 8, 8, 8)
    w3 = ox.randn(4, 2, 3, 3, 3)
    y3 = F.conv3d(x3, w3, padding=1)
    assert y3.shape == (1, 4, 8, 8, 8)

    # ConvTranspose2D
    yt = F.conv_transpose2d(y2, w2, padding=1)
    assert yt.shape == (2, 3, 16, 16)


def test_pooling_suite():
    x = ox.randn(2, 4, 16, 16)
    mp = F.max_pool2d(x, kernel_size=2, stride=2)
    assert mp.shape == (2, 4, 8, 8)

    ap = F.avg_pool2d(x, kernel_size=2, stride=2)
    assert ap.shape == (2, 4, 8, 8)

    adap = F.adaptive_avg_pool2d(x, (1, 1))
    assert adap.shape == (2, 4, 1, 1)

    adap_rect = F.adaptive_avg_pool2d(x, (4, 4))
    assert adap_rect.shape == (2, 4, 4, 4)

    # 1D pooling
    x1 = ox.randn(2, 4, 16)
    mp1 = F.max_pool1d(x1, kernel_size=2)
    assert mp1.shape == (2, 4, 8)
    ap1 = F.avg_pool1d(x1, kernel_size=2)
    assert ap1.shape == (2, 4, 8)


def test_normalizations():
    # BatchNorm 2D
    x = ox.randn(4, 8, 10, 10)
    rm = ox.zeros(8)
    rv = ox.ones(8)
    w = ox.ones(8)
    b = ox.zeros(8)
    bn = F.batch_norm(x, rm, rv, weight=w, bias=b, training=True)
    assert bn.shape == x.shape

    # LayerNorm
    ln = F.layer_norm(x, normalized_shape=[8, 10, 10])
    assert ln.shape == x.shape

    # GroupNorm
    gn = F.group_norm(x, num_groups=4)
    assert gn.shape == x.shape

    # InstanceNorm
    inn = F.instance_norm(x)
    assert inn.shape == x.shape


def test_spatial_and_vision_transforms():
    # PixelShuffle & PixelUnshuffle
    x = ox.randn(2, 16, 8, 8)
    shuffled = F.pixel_shuffle(x, upscale_factor=2)
    assert shuffled.shape == (2, 4, 16, 16)

    unshuffled = F.pixel_unshuffle(shuffled, downscale_factor=2)
    assert unshuffled.shape == (2, 16, 8, 8)

    # Interpolate
    interp = F.interpolate(x, size=(16, 16), mode="nearest")
    assert interp.shape == (2, 16, 16, 16)

    interp_bilinear = F.interpolate(x, size=(16, 16), mode="bilinear")
    assert interp_bilinear.shape == (2, 16, 16, 16)

    # Padding
    padded = F.pad(x, (1, 1, 2, 2))  # left=1, right=1, top=2, bottom=2
    assert padded.shape == (2, 16, 12, 10)

    # Unfold & Fold
    unfolded = F.unfold(x, kernel_size=3, padding=1)
    # L = 8 * 8 = 64, C_in * kH * kW = 16 * 9 = 144
    assert unfolded.shape == (2, 144, 64)
    folded = F.fold(unfolded, output_size=(8, 8), kernel_size=3, padding=1)
    assert folded.shape == (2, 16, 8, 8)


def test_embeddings_and_one_hot():
    weights = ox.randn(10, 32)
    indices = ox.tensor([[1, 2, 4], [0, 9, 3]])
    emb = F.embedding(indices, weights)
    assert emb.shape == (2, 3, 32)

    # EmbeddingBag
    offsets = ox.tensor([0, 3])
    bag = F.embedding_bag(indices.reshape(-1), weights, offsets=offsets, mode="mean")
    assert bag.shape == (2, 32)

    # One-hot
    idx = ox.tensor([0, 2, 1])
    oh = F.one_hot(idx, num_classes=5)
    assert oh.shape == (3, 5)
    assert np.allclose(oh.to_numpy(), [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0]])


def test_extended_activations():
    x = ox.tensor([-2.0, 0.0, 2.0])

    # GLU
    x_glu = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    glu_out = F.glu(x_glu, dim=-1)
    assert glu_out.shape == (2, 1)

    # Hardswish & Hardsigmoid
    hs = F.hardswish(x)
    assert hs.shape == (3,)
    hsig = F.hardsigmoid(x)
    assert hsig.shape == (3,)

    # Mish
    mish_out = F.mish(x)
    assert mish_out.shape == (3,)

    # ELU, SELU, CELU
    elu_out = F.elu(x)
    assert elu_out.shape == (3,)
    selu_out = F.selu(x)
    assert selu_out.shape == (3,)
    celu_out = F.celu(x)
    assert celu_out.shape == (3,)


def test_losses_suite():
    pred = ox.tensor([1.0, 2.0, 3.0])
    target = ox.tensor([1.2, 1.9, 3.1])

    # Smooth L1
    sl1 = F.smooth_l1_loss(pred, target)
    assert sl1.item() < 0.1

    # Poisson NLL
    pnll = F.poisson_nll_loss(pred, target)
    assert not math.isnan(pnll.item())

    # Gaussian NLL
    var = ox.tensor([0.1, 0.1, 0.1])
    gnll = F.gaussian_nll_loss(pred, target, var)
    assert not math.isnan(gnll.item())

    # Pairwise distance
    pdist = F.pairwise_distance(ox.tensor([[0.0, 0.0]]), ox.tensor([[3.0, 4.0]]))
    assert math.isclose(pdist.item(), 5.0, rel_tol=1e-4)
