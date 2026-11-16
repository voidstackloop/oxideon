"""Unit tests for ox.data (Dataset, DataLoader), ox.optim.lr_scheduler, callbacks, and new nn activations."""

import math
import numpy as np
import pytest
import oxideon as ox
import oxideon.nn as nn
from oxideon.data import Dataset, TensorDataset, DataLoader, random_split
from oxideon.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
from oxideon.callbacks import EarlyStopping


# ---------------------------------------------------------------------------
# 1. Dataset & DataLoader Tests
# ---------------------------------------------------------------------------

def test_tensor_dataset_indexing():
    x = ox.randn([10, 4])
    y = ox.zeros([10])
    ds = TensorDataset(x, y)

    assert len(ds) == 10
    sample_x, sample_y = ds[0]
    assert sample_x.shape == [4]
    assert sample_y.shape == []


def test_dataloader_batching_and_collation():
    x = ox.randn([100, 8])
    y = ox.ones([100])
    ds = TensorDataset(x, y)

    loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=False, seed=42)
    assert len(loader) == 7  # ceil(100 / 16) = 7

    total_samples = 0
    for batch_x, batch_y in loader:
        assert isinstance(batch_x, ox.Tensor)
        assert isinstance(batch_y, ox.Tensor)
        assert batch_x.shape[1] == 8
        total_samples += batch_x.shape[0]

    assert total_samples == 100

    # With drop_last=True
    loader_drop = DataLoader(ds, batch_size=16, drop_last=True)
    assert len(loader_drop) == 6  # 100 // 16 = 6
    total_drop_samples = sum(bx.shape[0] for bx, _ in loader_drop)
    assert total_drop_samples == 96


def test_random_split_lengths():
    x = ox.randn([50, 4])
    ds = TensorDataset(x)
    train_ds, val_ds, test_ds = random_split(ds, [30, 10, 10], seed=42)

    assert len(train_ds) == 30
    assert len(val_ds) == 10
    assert len(test_ds) == 10


# ---------------------------------------------------------------------------
# 2. Learning Rate Scheduler Tests
# ---------------------------------------------------------------------------

def test_step_lr_scheduler():
    w = ox.randn([2, 2]).requires_grad_(True)
    opt = ox.optim.Adam([w], lr=0.1)
    sched = StepLR(opt, step_size=2, gamma=0.5)

    assert math.isclose(sched.get_last_lr()[0], 0.1, abs_tol=1e-5)
    sched.step()  # epoch 1
    assert math.isclose(sched.get_last_lr()[0], 0.1, abs_tol=1e-5)
    sched.step()  # epoch 2 (decay)
    assert math.isclose(sched.get_last_lr()[0], 0.05, abs_tol=1e-5)
    sched.step()  # epoch 3
    assert math.isclose(sched.get_last_lr()[0], 0.05, abs_tol=1e-5)
    sched.step()  # epoch 4 (decay)
    assert math.isclose(sched.get_last_lr()[0], 0.025, abs_tol=1e-5)


def test_cosine_annealing_lr():
    w = ox.randn([2, 2]).requires_grad_(True)
    opt = ox.optim.Adam([w], lr=0.1)
    sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.01)

    lrs = []
    for _ in range(10):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # Should monotonically decrease to eta_min
    assert lrs[0] > lrs[-1]
    assert math.isclose(lrs[-1], 0.01, abs_tol=1e-4)


def test_one_cycle_lr():
    w = ox.randn([2, 2]).requires_grad_(True)
    opt = ox.optim.Adam([w], lr=0.01)
    sched = OneCycleLR(opt, max_lr=0.1, total_steps=20, pct_start=0.3)

    lrs = []
    for _ in range(20):
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # Should increase up to max_lr around step 6, then decrease
    peak_idx = int(np.argmax(lrs))
    assert 4 <= peak_idx <= 8
    assert lrs[peak_idx] > lrs[0]
    assert lrs[-1] < lrs[peak_idx]


def test_reduce_lr_on_plateau():
    w = ox.randn([2, 2]).requires_grad_(True)
    opt = ox.optim.Adam([w], lr=0.1)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    # Loss is decreasing
    sched.step(1.0)
    sched.step(0.8)
    sched.step(0.6)
    assert math.isclose(opt.lr, 0.1, abs_tol=1e-5)

    # Loss plateaus for 3 epochs (patience=2 exceeded)
    sched.step(0.60001)
    sched.step(0.60002)
    sched.step(0.60003)
    assert math.isclose(opt.lr, 0.05, abs_tol=1e-5)


def _adam(lr):
    w = ox.randn([2, 2]).requires_grad_(True)
    return ox.optim.Adam([w], lr=lr)


def test_cosine_annealing_lr_matches_torch_past_tmax():
    # torch CosineAnnealingLR(T_max=4, eta_min=0.01): oscillates past T_max
    opt = _adam(0.1)
    sched = CosineAnnealingLR(opt, T_max=4, eta_min=0.01)
    assert math.isclose(sched.get_last_lr()[0], 0.1, abs_tol=1e-9)
    want = [0.086819805153, 0.055, 0.023180194847, 0.01, 0.023180194847, 0.055, 0.086819805153]
    for expected in want:
        sched.step()
        assert math.isclose(sched.get_last_lr()[0], expected, abs_tol=1e-9)


def test_one_cycle_lr_matches_torch():
    # torch OneCycleLR(max=0.1, total=20, pct=0.3): linear ramp peaks at
    # epoch 5, cosine tail, rises again at epoch 20, raises past total
    opt = _adam(0.01)
    sched = OneCycleLR(opt, max_lr=0.1, total_steps=20, pct_start=0.3)
    assert math.isclose(sched.get_last_lr()[0], 0.004, abs_tol=1e-9)
    want = [
        0.01316718427, 0.03716718427, 0.06683281573, 0.09083281573, 0.1,
        0.098746400624, 0.095048463201, 0.089091617757, 0.081174565395,
        0.071694300179, 0.061126202194, 0.0500002, 0.038874197806,
        0.028306099821, 0.018825834605, 0.010908782243, 0.004951936799,
        0.001253999376, 0.0000004, 0.001253999376,
    ]
    for expected in want:
        sched.step()
        assert math.isclose(sched.get_last_lr()[0], expected, abs_tol=1e-9)
    with pytest.raises(ValueError, match="total steps"):
        sched.step()


def test_exponential_lr_chains_from_current_lr():
    # torch ExponentialLR compounds the live lr even after external edits
    opt = _adam(0.1)
    sched = ExponentialLR(opt, gamma=0.9)
    assert math.isclose(sched.get_last_lr()[0], 0.1, abs_tol=1e-12)
    sched.step()
    assert math.isclose(sched.get_last_lr()[0], 0.09, abs_tol=1e-12)
    sched.step()
    assert math.isclose(sched.get_last_lr()[0], 0.081, abs_tol=1e-12)
    opt.set_learning_rate(0.5)
    sched.step()
    assert math.isclose(sched.get_last_lr()[0], 0.45, abs_tol=1e-12)


def test_plateau_relative_threshold_matches_torch():
    # torch default threshold_mode='rel': metric must beat best*(1-1e-4)
    opt = _adam(0.1)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    want = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.025]
    for metric, expected in zip([1.0, 0.99999, 0.99998, 0.99997, 0.5, 0.5, 0.5, 0.5], want):
        sched.step(metric)
        assert math.isclose(opt.lr, expected, abs_tol=1e-9)


def test_plateau_threshold_mode_rel_vs_abs():
    # 0.00095 vs best 0.001: 5% better (rel) but within 1e-4 absolute (abs)
    opt = _adam(0.1)
    rel = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=0)
    rel.step(0.001)
    rel.step(0.00095)
    assert math.isclose(opt.lr, 0.1, abs_tol=1e-12)
    opt2 = _adam(0.1)
    absolute = ReduceLROnPlateau(opt2, mode="min", factor=0.5, patience=0, threshold_mode="abs")
    absolute.step(0.001)
    absolute.step(0.00095)
    assert math.isclose(opt2.lr, 0.05, abs_tol=1e-12)
    with pytest.raises(ValueError, match="threshold_mode"):
        ReduceLROnPlateau(_adam(0.1), threshold_mode="bogus")


# ---------------------------------------------------------------------------
# 3. Early Stopping Callback Test
# ---------------------------------------------------------------------------

def test_early_stopping_callback():
    model = nn.Sequential(nn.Linear(2, 2))
    es = EarlyStopping(patience=3, min_delta=1e-4, mode="min", restore_best_weights=True)

    # Decreasing loss
    assert not es(1.0, model)
    assert not es(0.8, model)
    assert not es(0.6, model)

    # Stagnant loss
    assert not es(0.61, model)
    assert not es(0.62, model)
    should_stop = es(0.63, model)
    assert should_stop is True


# ---------------------------------------------------------------------------
# 4. Neural Network Activation & Attention Layers
# ---------------------------------------------------------------------------

def test_nn_activations():
    x = ox.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))

    gelu = nn.GELU()
    assert gelu(x).shape == [3]

    silu = nn.SiLU()
    assert silu(x).shape == [3]

    sig = nn.Sigmoid()
    assert sig(x).shape == [3]
    np.testing.assert_allclose(sig(x).to_numpy(), [0.26894, 0.5, 0.731058], atol=1e-4)

    tanh = nn.Tanh()
    assert tanh(x).shape == [3]

    leaky = nn.LeakyReLU(negative_slope=0.1)
    assert leaky(x).shape == [3]
    np.testing.assert_allclose(leaky(x).to_numpy(), [-0.1, 0.0, 1.0], atol=1e-4)

    soft = nn.Softmax(dim=0)
    assert math.isclose(np.sum(soft(x).to_numpy()), 1.0, abs_tol=1e-5)


def test_multihead_attention():
    mha = nn.MultiheadAttention(embed_dim=16, num_heads=4)
    x = ox.randn([8, 16])

    out, attn = mha(x)
    assert out.shape == [8, 16]
    assert attn.shape == [8, 8]
