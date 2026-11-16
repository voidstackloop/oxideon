"""
Unit tests for Ultra-Low-Bit K-Quants (Q2_K, Q3_K, Q5_K, Q6_K) and 1.58-bit BitNet in Oxideon:
- Q2_K (2-bit packed, 4 values/byte)
- Q3_K (3-bit, 8 levels)
- Q5_K (5-bit, 32 levels)
- Q6_K (6-bit, 64 levels)
- Unified quantize_k_quant and dequantize_k_quant
- BitNet 1.58b ternary quantization {-1, 0, 1} and BitLinear
"""
import numpy as np
import pytest
import oxideon as ox
from oxideon import kquants

def test_q2_k_quantization():
    w = ox.randn([32, 32])
    block = kquants.Q2KQuantizer.quantize(w, block_size=32)
    assert block.quant_type == "q2_k"
    assert block.bits == 2
    # Total elements = 1024, packed at 4 per byte => 256 bytes!
    assert len(block.data) == 256

    recon = kquants.Q2KQuantizer.dequantize(block)
    assert recon.shape == [32, 32]
    # Check positive correlation with original weights
    corr = np.corrcoef(w.to_numpy().flatten(), recon.to_numpy().flatten())[0, 1]
    assert corr > 0.85

def test_q3_k_quantization():
    w = ox.randn([16, 32])
    block = kquants.Q3KQuantizer.quantize(w, block_size=32)
    assert block.quant_type == "q3_k"
    assert block.bits == 3
    assert np.all(block.data >= 0) and np.all(block.data <= 7)

    recon = kquants.Q3KQuantizer.dequantize(block)
    assert recon.shape == [16, 32]
    corr = np.corrcoef(w.to_numpy().flatten(), recon.to_numpy().flatten())[0, 1]
    assert corr > 0.92

def test_q5_k_quantization():
    w = ox.randn([16, 32])
    block = kquants.Q5KQuantizer.quantize(w, block_size=32)
    assert block.quant_type == "q5_k"
    assert block.bits == 5
    assert np.all(block.data >= 0) and np.all(block.data <= 31)

    recon = kquants.Q5KQuantizer.dequantize(block)
    assert recon.shape == [16, 32]
    corr = np.corrcoef(w.to_numpy().flatten(), recon.to_numpy().flatten())[0, 1]
    assert corr > 0.985

def test_q6_k_quantization():
    w = ox.randn([16, 32])
    block = kquants.Q6KQuantizer.quantize(w, block_size=32)
    assert block.quant_type == "q6_k"
    assert block.bits == 6
    assert np.all(block.data >= 0) and np.all(block.data <= 63)

    recon = kquants.Q6KQuantizer.dequantize(block)
    assert recon.shape == [16, 32]
    corr = np.corrcoef(w.to_numpy().flatten(), recon.to_numpy().flatten())[0, 1]
    assert corr > 0.995

def test_unified_k_quant_dispatch():
    w = ox.randn([8, 16])
    for q_type in ["q2_k", "q3_k", "q5_k", "q6_k", "q2", "q3", "q5", "q6"]:
        b = ox.quantize_k_quant(w, quant_type=q_type, block_size=16)
        r = ox.dequantize_k_quant(b)
        assert r.shape == [8, 16]

    with pytest.raises(ValueError):
        ox.quantize_k_quant(w, quant_type="q99_invalid")

def test_ternary_bitnet_quantization_and_bitlinear():
    w = ox.from_numpy(np.array([-0.8, -0.4, 0.05, 0.35, 0.9], dtype=np.float32))
    w_tern, gamma = kquants.quantize_ternary_bitnet(w)
    unique_vals = set(w_tern.to_numpy().flatten())
    # All values must be in {-1, 0, 1}
    assert unique_vals.issubset({-1.0, 0.0, 1.0})
    assert gamma > 0.0

    recon = kquants.dequantize_ternary_bitnet(w_tern, gamma)
    assert recon.shape == [5]

    # BitLinear forward pass
    bit_linear = kquants.BitLinear(in_features=8, out_features=4, bias=True)
    x = ox.randn([3, 8])
    out = bit_linear(x)
    assert out.shape == [3, 4]
