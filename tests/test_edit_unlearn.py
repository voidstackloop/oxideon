"""Tests for knowledge editing and unlearning (oxideon.edit)."""
import numpy as np
import pytest

import oxideon as ox
from oxideon import GALoss, NPOLoss, memit_layer_update, rome_update


def t(arr):
    return ox.from_numpy(np.asarray(arr, dtype=np.float32))


def test_top_level_exports():
    assert ox.rome_update is rome_update
    assert ox.NPOLoss is NPOLoss
    assert ox.edit.GALoss is GALoss


def test_rome_exact_memorization():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 6)).astype(np.float32)
    k = rng.normal(size=6)
    v = rng.normal(size=4)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T + np.eye(6)
    before = w.copy()
    out = rome_update(t(w), k, v, cov).to_numpy()
    assert out.shape == (4, 6)
    assert np.allclose(out @ k, v, atol=1e-5)
    assert np.array_equal(w, before)


def test_rome_validation():
    w = t(np.eye(3, dtype=np.float32))
    k = np.ones(3)
    v = np.zeros(3)
    cov = np.eye(3)
    with pytest.raises(ValueError):
        rome_update(w, k, v, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        rome_update(w, np.zeros(3), v, cov)
    with pytest.raises(ValueError):
        rome_update(w, np.ones(2), v, cov)
    with pytest.raises(ValueError):
        rome_update(w, k, np.zeros(2), cov)
    with pytest.raises(ValueError):
        rome_update(w, k, v, np.eye(2))
    with pytest.raises(ValueError):
        rome_update(t(np.ones(3, dtype=np.float32)), k, v, cov)
    bad = t(np.eye(3, dtype=np.float32))
    bad_np = bad.to_numpy().copy()
    bad_np[0, 0] = np.nan
    with pytest.raises(ValueError):
        rome_update(t(bad_np), k, v, cov)


def test_memit_exact_with_square_keys():
    keys = np.array([[1.0, 0.2, 0.1], [0.1, 1.0, 0.2], [0.05, 0.1, 1.0]])
    vals = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0]])
    w = np.zeros((2, 3), dtype=np.float32)
    out = memit_layer_update(t(w), keys, vals, np.zeros((3, 3))).to_numpy()
    assert out.shape == (2, 3)
    assert np.allclose(out @ keys.T, vals.T, atol=1e-4)


def test_memit_decreases_residual_and_handles_singular():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(3, 5)).astype(np.float32)
    keys = rng.normal(size=(4, 5))
    vals = rng.normal(size=(4, 3))
    before = float(np.linalg.norm(vals.T - w @ keys.T))
    out = memit_layer_update(t(w), keys, vals, np.eye(5)).to_numpy()
    after = float(np.linalg.norm(vals.T - out @ keys.T))
    assert after < before
    single = memit_layer_update(t(w), keys[:1], vals[:1], np.eye(5), layers_remaining=2)
    assert single.to_numpy().shape == (3, 5)
    with pytest.raises(ValueError):
        memit_layer_update(t(w), keys[:1], vals[:1], np.zeros((5, 5)))
    damped = memit_layer_update(t(w), keys[:1], vals[:1], np.zeros((5, 5)), damping=0.1)
    assert damped.to_numpy().shape == (3, 5)


def test_memit_validation():
    w = t(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        memit_layer_update(w, np.zeros((0, 3)), np.zeros((0, 2)), np.eye(3))
    with pytest.raises(ValueError):
        memit_layer_update(w, np.zeros((2, 3)), np.zeros((2, 3)), np.eye(3))
    with pytest.raises(ValueError):
        memit_layer_update(w, np.zeros((2, 3)), np.zeros((2, 2)), np.eye(3), layers_remaining=0)
    with pytest.raises(ValueError):
        memit_layer_update(w, np.zeros((2, 3)), np.zeros((2, 2)), np.eye(3), damping=-1.0)


def test_npo_hand_computed():
    loss, metrics = NPOLoss(beta=1.0)(t([0.0]), t([0.0]))
    assert loss.to_numpy().tolist() == pytest.approx([0.6931472], abs=1e-6)
    loss, metrics2 = NPOLoss(beta=1.0)(t([1.0]), t([-1.0]))
    assert loss.to_numpy().tolist() == pytest.approx([2.126928], abs=1e-5)
    assert metrics2["mean_log_ratio"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        NPOLoss(beta=0.0)
    with pytest.raises(ValueError):
        NPOLoss()(t([0.0]), t([0.0, 0.0]))
    with pytest.raises(ValueError):
        NPOLoss()(t([]), t([]))


def test_ga_hand_computed():
    loss, metrics = GALoss()(t([-1.0, -3.0]))
    assert loss.to_numpy().tolist() == [-2.0]
    assert metrics["mean_forget_logp"] == -2.0
    loss, metrics = GALoss(retain_weight=1.0)(t([-1.0, -3.0]), t([-0.5]))
    assert loss.to_numpy().tolist() == [-1.5]
    assert metrics["mean_retain_logp"] == -0.5
    with pytest.raises(ValueError):
        GALoss(retain_weight=-1.0)
    with pytest.raises(ValueError):
        GALoss()(t([]))
