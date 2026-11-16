"""Tests for KV eviction, MXFP4, MLA and NSA (inference efficiency)."""
import numpy as np
import pytest

import oxideon as ox
from oxideon import (
    MLAAttention,
    MXFP4_LEVELS,
    NativeSparseAttention,
    ScoredEvictionCache,
    StreamingSinkCache,
    dequantize_mxfp4,
    quantize_mxfp4,
)


def _kv(positions, heads=2, dim=4, batch=1):
    n = len(positions)
    k = np.zeros((batch, n, heads, dim), np.float32)
    v = np.zeros((batch, n, heads, dim), np.float32)
    for j, p in enumerate(positions):
        k[:, j] = p
        v[:, j] = -p
    return ox.from_numpy(k), ox.from_numpy(v)


def _kept(k_out):
    return [int(x) for x in k_out.to_numpy()[0, :, 0, 0]]


def _kept_v(v_out):
    return [int(x) for x in v_out.to_numpy()[0, :, 0, 0]]


def test_sink_cache_keeps_first_and_last():
    cache = StreamingSinkCache(1, 8, 2, 4, n_sink=2)
    k, v = _kv([0, 1, 2, 3])
    ko, _ = cache.update(k, v)
    assert _kept(ko) == [0, 1, 2, 3]
    k, v = _kv([4, 5, 6, 7])
    ko, _ = cache.update(k, v)
    assert _kept(ko) == [0, 1, 2, 3, 4, 5, 6, 7]
    k, v = _kv([8, 9, 10, 11])
    ko, vo = cache.update(k, v)
    assert _kept(ko) == [0, 1, 6, 7, 8, 9, 10, 11]
    assert _kept_v(vo) == [0, -1, -6, -7, -8, -9, -10, -11]
    cache.reset()
    k, v = _kv([0, 1])
    ko, _ = cache.update(k, v)
    assert _kept(ko) == [0, 1]


def test_sink_cache_validation():
    with pytest.raises(ValueError):
        StreamingSinkCache(1, 8, 2, 4, n_sink=8)
    cache = StreamingSinkCache(1, 8, 2, 4)
    k, v = _kv([0, 1], batch=2)
    with pytest.raises(ValueError):
        cache.update(k, v)


def test_h2o_keeps_heavy_hitters():
    cache = ScoredEvictionCache(1, 6, 1, 4, recent_window=2, policy="h2o")
    k, v = _kv([0, 1, 2, 3], heads=1)
    cache.update(k, v, np.zeros((1, 1, 4, 4)))
    k, v = _kv([4, 5, 6, 7], heads=1)
    scores = np.zeros((1, 1, 4, 8))
    scores[0, 0, :, 1] = 10.0
    ko, _ = cache.update(k, v, scores)
    assert _kept(ko) == [0, 1, 2, 3, 6, 7]


def test_tova_uses_last_query_scores():
    cache = ScoredEvictionCache(1, 6, 1, 4, recent_window=2, policy="tova")
    k, v = _kv([0, 1, 2, 3], heads=1)
    cache.update(k, v, np.zeros((1, 1, 4, 4)))
    k, v = _kv([4, 5, 6, 7], heads=1)
    scores = np.zeros((1, 1, 4, 8))
    scores[0, 0, :, 0] = 10.0  # early queries love position 0 ...
    scores[0, 0, -1, :] = 0.0
    scores[0, 0, -1, 4] = 10.0  # ... but the last query loves position 4
    ko, _ = cache.update(k, v, scores)
    kept = _kept(ko)
    assert 4 in kept and 6 in kept and 7 in kept and len(kept) == 6


def test_snap_pools_observe_window():
    cache = ScoredEvictionCache(
        1, 6, 1, 4, recent_window=2, policy="snap", observe_window=2, pool_size=2
    )
    k, v = _kv([0, 1, 2, 3], heads=1)
    cache.update(k, v, np.zeros((1, 1, 4, 4)))
    k, v = _kv([4, 5, 6, 7], heads=1)
    scores = np.zeros((1, 1, 4, 8))
    scores[0, 0, -2:, 5] = 10.0
    ko, _ = cache.update(k, v, scores)
    kept = _kept(ko)
    assert 5 in kept and len(kept) == 6
    cache.reset()
    assert cache.cur_len == 0


def test_scored_cache_validation():
    with pytest.raises(ValueError):
        ScoredEvictionCache(1, 8, 2, 4, policy="mru")
    with pytest.raises(ValueError):
        ScoredEvictionCache(1, 8, 2, 4, recent_window=8)
    cache = ScoredEvictionCache(1, 6, 2, 4, recent_window=2)
    k, v = _kv([0, 1])
    with pytest.raises(ValueError):
        cache.update(k, v, np.zeros((1, 2, 2, 3)))
    k2, v2 = _kv([0, 1], batch=2)
    with pytest.raises(ValueError):
        cache.update(k2, v2, np.zeros((2, 2, 2, 2)))


def test_mxfp4_levels_and_exact_codes():
    assert MXFP4_LEVELS == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    x = ox.from_numpy(np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -3], dtype=np.float32))
    codes, scales = quantize_mxfp4(x)
    assert codes.to_numpy().flatten().tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 13]
    assert scales.to_numpy().tolist() == [1.0]
    back = dequantize_mxfp4(codes, scales).to_numpy().flatten().tolist()
    assert back == [0, 0.5, 1, 1.5, 2, 3, 4, 6, -3]


def test_mxfp4_zeros_padding_and_roundtrip():
    codes, scales = quantize_mxfp4(ox.from_numpy(np.zeros(10, dtype=np.float32)))
    assert (codes.to_numpy() == 0).all()
    assert (dequantize_mxfp4(codes, scales).to_numpy() == 0).all()
    rng = np.random.default_rng(0)
    w = (rng.normal(0, 0.5, size=(5, 37))).astype(np.float32)
    codes, scales = quantize_mxfp4(ox.from_numpy(w))
    assert list(codes.shape) == [5, 37]
    assert np.all(np.log2(scales.to_numpy()) == np.round(np.log2(scales.to_numpy())))
    back = dequantize_mxfp4(codes, scales).to_numpy()
    err = np.abs(back - w)
    assert err.mean() < 0.06
    assert err.max() < 0.6


def test_mxfp4_nonfinite_and_validation():
    bad = ox.from_numpy(np.array([1.0, float("nan")], dtype=np.float32))
    with pytest.raises(ValueError):
        quantize_mxfp4(bad)
    inf = ox.from_numpy(np.array([1.0, float("inf"), float("-inf")], dtype=np.float32))
    codes, scales = quantize_mxfp4(inf)
    assert np.all(np.isfinite(dequantize_mxfp4(codes, scales).to_numpy()))
    good, scales = quantize_mxfp4(ox.from_numpy(np.ones(4, dtype=np.float32)))
    broken = good.to_numpy().copy()
    broken[0] = 16
    with pytest.raises(ValueError):
        dequantize_mxfp4(ox.from_numpy(broken), scales)
    with pytest.raises(ValueError):
        dequantize_mxfp4(good, ox.from_numpy(np.ones(7, dtype=np.float32)))
    with pytest.raises(ValueError):
        quantize_mxfp4(ox.from_numpy(np.ones(4, dtype=np.float32)), block_size=0)


def test_mla_shapes_determinism_causality():
    attn = MLAAttention(64, 4, q_lora_rank=8, kv_lora_rank=8)
    rng = np.random.default_rng(1)
    x = ox.from_numpy(rng.normal(size=(2, 16, 64)).astype(np.float32))
    out = attn(x).to_numpy()
    assert out.shape == (2, 16, 64)
    assert np.all(np.isfinite(out))
    assert np.array_equal(out, attn(x).to_numpy())
    assert attn(x, causal=False).to_numpy().shape == (2, 16, 64)
    pert = x.to_numpy().copy()
    pert[:, 8:] += 5.0
    out2 = attn(ox.from_numpy(pert)).to_numpy()
    assert np.allclose(out[:, :8], out2[:, :8], atol=1e-5)


def test_mla_validation():
    with pytest.raises(ValueError):
        MLAAttention(64, 4, qk_rope_head_dim=15)
    attn = MLAAttention(64, 4)
    with pytest.raises(ValueError):
        attn(ox.from_numpy(np.zeros((1, 4, 32), dtype=np.float32)))


def test_nsa_shapes_determinism_causality():
    attn = NativeSparseAttention(32, 4, block_size=8, top_blocks=2, window_size=8)
    rng = np.random.default_rng(2)
    x = ox.from_numpy(rng.normal(size=(2, 24, 32)).astype(np.float32))
    out = attn(x).to_numpy()
    assert out.shape == (2, 24, 32)
    assert np.all(np.isfinite(out))
    assert np.array_equal(out, attn(x).to_numpy())
    pert = x.to_numpy().copy()
    pert[:, 12:] += 5.0
    out2 = attn(ox.from_numpy(pert)).to_numpy()
    assert np.allclose(out[:, :12], out2[:, :12], atol=1e-5)


def test_nsa_edges_and_validation():
    tiny = NativeSparseAttention(16, 2, block_size=8, top_blocks=99, window_size=4)
    x = ox.from_numpy(np.random.default_rng(3).normal(size=(1, 4, 16)).astype(np.float32))
    out = tiny(x).to_numpy()
    assert out.shape == (1, 4, 16)
    assert np.all(np.isfinite(out))
    unit = NativeSparseAttention(16, 2, block_size=1, top_blocks=1, window_size=2)
    assert unit(x).to_numpy().shape == (1, 4, 16)
    with pytest.raises(ValueError):
        NativeSparseAttention(30, 4)
    with pytest.raises(ValueError):
        tiny(ox.from_numpy(np.zeros((1, 4, 8), dtype=np.float32)))
