"""Test suite for Oxideon Quantization Enrichment:
  - Per-tensor and Per-channel affine quantization & dequantization
  - Dynamic per-channel scale/zero-point calibration
  - PerChannelQuantizedLinear & QuantizedLinear
  - QuantizedEmbedding with on-demand row dequantization
  - SmoothQuant outlier migration & equivalence
  - Scaled FP8 GEMM & FP8Linear (E4M3 and E5M2)
  - Recursive quantize_dynamic with multi-policy support
"""

import numpy as np
import pytest
import oxideon as ox
from oxideon.quantization import (
    quantize_per_tensor,
    dequantize,
    quantize_per_channel,
    dequantize_per_channel,
    quantize_per_channel_dynamic,
    QuantizedLinear,
    PerChannelQuantizedLinear,
    QuantizedEmbedding,
    smooth_quant,
    fp8_gemm,
    FP8Linear,
    quantize_dynamic,
)
from oxideon.nn import Module, Linear, Embedding


def test_per_channel_quant_roundtrip():
    # Shape: [4, 8]
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 8)).astype(np.float32)

    # Dynamic symmetric per-channel along axis 0 (rows)
    q_t, scales, zps = quantize_per_channel_dynamic(x, axis=0, dtype="qint8", symmetric=True)
    assert len(scales) == 4
    assert len(zps) == 4
    assert np.all(zps == 0)

    deq = dequantize_per_channel(q_t, scales, zps, axis=0)
    deq_np = deq.to_numpy()
    assert deq_np.shape == (4, 8)
    np.testing.assert_allclose(deq_np, x, atol=0.08)

    # Dynamic asymmetric per-channel along axis 1 (cols)
    q_t1, scales1, zps1 = quantize_per_channel_dynamic(x, axis=1, dtype="quint8", symmetric=False)
    assert len(scales1) == 8
    assert len(zps1) == 8
    deq1 = dequantize_per_channel(q_t1, scales1, zps1, axis=1)
    np.testing.assert_allclose(deq1.to_numpy(), x, atol=0.08)


def test_per_channel_quantized_linear():
    lin = Linear(8, 4, bias=True)
    # Set known weights and bias
    rng = np.random.default_rng(101)
    w = rng.uniform(-1.0, 1.0, (4, 8)).astype(np.float32)
    b = rng.uniform(-0.5, 0.5, (4,)).astype(np.float32)
    lin.weight = ox.nn.Parameter(ox.from_numpy(w))
    lin.bias = ox.nn.Parameter(ox.from_numpy(b))

    qlin = PerChannelQuantizedLinear.from_float(lin, dtype="qint8", symmetric=True)
    assert qlin.in_features == 8
    assert qlin.out_features == 4
    assert qlin.qweight.shape == (4, 8)
    assert len(qlin.scales) == 4

    x = ox.from_numpy(rng.standard_normal((2, 8)).astype(np.float32))
    y_float = lin(x).to_numpy()
    y_quant = qlin(x).to_numpy()
    np.testing.assert_allclose(y_quant, y_float, atol=0.05)


def test_quantized_embedding():
    num_emb = 16
    emb_dim = 8
    emb = Embedding(num_emb, emb_dim)
    q_emb = QuantizedEmbedding.from_float(emb, dtype="qint8", symmetric=True)

    assert q_emb.num_embeddings == num_emb
    assert q_emb.embedding_dim == emb_dim
    assert q_emb.qweight.shape == (num_emb, emb_dim)

    # Fetch token indices
    indices = ox.from_numpy(np.array([[1, 5, 2], [0, 15, 3]], dtype=np.int64))
    float_out = emb(indices).to_numpy()
    quant_out = q_emb(indices).to_numpy()

    assert quant_out.shape == (2, 3, 8)
    np.testing.assert_allclose(quant_out, float_out, atol=0.05)


def test_smooth_quant_equivalence():
    in_dim = 6
    out_dim = 4
    lin = Linear(in_dim, out_dim, bias=True)

    rng = np.random.default_rng(202)
    w = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    b = rng.standard_normal((out_dim,)).astype(np.float32)
    lin.weight = ox.nn.Parameter(ox.from_numpy(w))
    lin.bias = ox.nn.Parameter(ox.from_numpy(b))

    # Channel-wise activation maximums with an outlier channel (channel 2)
    act_max = np.array([1.2, 0.8, 45.0, 1.1, 0.9, 1.5], dtype=np.float32)

    smoothed_lin, scales = smooth_quant(lin, act_max, alpha=0.5)
    s_np = scales.to_numpy()
    assert s_np.shape == (in_dim,)

    # Outlier channel 2 should have significantly higher smoothing scale
    assert s_np[2] > s_np[0]

    # Verify mathematical equivalence: (X / s) @ W_smoothed^T + b == X @ W^T + b
    x_np = rng.standard_normal((3, in_dim)).astype(np.float32)
    x = ox.from_numpy(x_np)
    orig_y = lin(x).to_numpy()

    x_smoothed = ox.from_numpy(x_np / s_np)
    smoothed_y = smoothed_lin(x_smoothed).to_numpy()
    np.testing.assert_allclose(smoothed_y, orig_y, rtol=1e-5, atol=1e-5)


def test_fp8_gemm_and_fp8_linear():
    rng = np.random.default_rng(77)
    a = ox.from_numpy(rng.uniform(-2.0, 2.0, (4, 8)).astype(np.float32))
    b = ox.from_numpy(rng.uniform(-2.0, 2.0, (8, 6)).astype(np.float32))

    # Scaled GEMM
    out_e4m3 = fp8_gemm(a, b, scale_a=1.0, scale_b=1.0, format="e4m3")
    assert out_e4m3.shape == [4, 6]
    expected = a.to_numpy() @ b.to_numpy()
    np.testing.assert_allclose(out_e4m3.to_numpy(), expected, atol=0.5)

    # FP8Linear
    lin = Linear(8, 4, bias=True)
    fp8_lin = FP8Linear.from_float(lin, format="e4m3")
    x = ox.from_numpy(rng.standard_normal((2, 8)).astype(np.float32))
    y_float = lin(x).to_numpy()
    y_fp8 = fp8_lin(x).to_numpy()
    np.testing.assert_allclose(y_fp8, y_float, atol=0.2)


def test_quantize_dynamic_model():
    class ToyModel(Module):
        def __init__(self):
            super().__init__()
            self.emb = Embedding(10, 8)
            self.fc1 = Linear(8, 16)
            self.fc2 = Linear(16, 4)

        def forward(self, idx):
            h = self.emb(idx)
            h = self.fc1(h)
            return self.fc2(h)

    # Test INT8 dynamic quantization with per-channel linear and quantized embedding
    m1 = ToyModel()
    quantize_dynamic(m1, dtype="qint8", per_channel=True, quantize_embeddings=True)
    assert isinstance(m1.fc1, PerChannelQuantizedLinear)
    assert isinstance(m1.fc2, PerChannelQuantizedLinear)
    assert isinstance(m1.emb, QuantizedEmbedding)

    idx = ox.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.int64))
    out1 = m1(idx)
    assert out1.shape == [2, 2, 4]

    # Test FP8 dynamic quantization
    m2 = ToyModel()
    quantize_dynamic(m2, dtype="fp8_e4m3", quantize_embeddings=False)
    assert isinstance(m2.fc1, FP8Linear)
    assert isinstance(m2.fc2, FP8Linear)
    assert isinstance(m2.emb, Embedding)
    out2 = m2(idx)
    assert out2.shape == [2, 2, 4]
