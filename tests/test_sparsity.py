"""Tests for Wanda / SparseGPT / N:M / depth pruning."""
import numpy as np
import pytest

import oxideon as ox
from oxideon import (
    block_influence,
    drop_sequential_blocks,
    get_pruning_mask,
    get_sparsity,
    has_pruning_mask,
    prune_nm_structured,
    prune_sparsegpt,
    prune_wanda,
    remove_pruning,
    select_layers_to_drop,
)


def linear(weight):
    arr = np.asarray(weight, dtype=np.float32)
    mod = ox.nn.Linear(arr.shape[1], arr.shape[0], bias=False)
    mod.weight = ox.nn.Parameter(ox.from_numpy(arr))
    return mod


def test_top_level_exports():
    assert ox.prune_wanda is prune_wanda
    assert ox.prune_sparsegpt is prune_sparsegpt
    assert ox.block_influence is block_influence


def test_wanda_metric_and_exact_counts():
    mod = linear([[1.0, 50.0, 60.0, 1.0]])
    acts = np.diag([100.0, 1.0, 1.0, 1.0])
    prune_wanda(mod, activations=acts, amount=0.5)
    assert mod.weight.to_numpy().tolist() == [[1.0, 0.0, 60.0, 0.0]]
    assert has_pruning_mask(mod)
    assert get_pruning_mask(mod).tolist() == [[1.0, 0.0, 1.0, 0.0]]
    assert get_sparsity(mod.weight) == 0.5
    keep = linear([[4.0, 3.0, 2.0, 1.0]])
    prune_wanda(keep, activations=np.ones((4, 4)), amount=0.0)
    assert keep.weight.to_numpy().tolist() == [[4.0, 3.0, 2.0, 1.0]]
    gone = linear([[4.0, 3.0, 2.0, 1.0]])
    prune_wanda(gone, activations=np.ones((4, 4)), amount=1.0)
    assert get_sparsity(gone.weight) == 1.0


def test_wanda_validation():
    mod = linear([[1.0, 2.0]])
    with pytest.raises(ValueError):
        prune_wanda(mod)
    with pytest.raises(ValueError):
        prune_wanda(mod, activations=np.ones((3, 3)), amount=0.5)
    with pytest.raises(ValueError):
        prune_wanda(mod, activations=np.ones((3, 2)), amount=1.5)
    bad = linear([[1.0, float("nan")]])
    with pytest.raises(ValueError):
        prune_wanda(bad, activations=np.ones((3, 2)))


def test_sparsegpt_sparsity_compensation_determinism():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 8)).astype(np.float32)
    x = rng.normal(size=(32, 8))
    mod = linear(w)
    prune_sparsegpt(mod, activations=x, amount=0.5, blocksize=4, damping=0.01)
    assert get_sparsity(mod.weight) == 0.5
    assert has_pruning_mask(mod)
    masked = w * get_pruning_mask(mod)
    assert not np.allclose(mod.weight.to_numpy(), masked)
    again = linear(w)
    prune_sparsegpt(again, activations=x, amount=0.5, blocksize=4, damping=0.01)
    assert np.array_equal(mod.weight.to_numpy(), again.weight.to_numpy())
    untouched = linear(w)
    prune_sparsegpt(untouched, activations=x, amount=0.0)
    assert np.array_equal(untouched.weight.to_numpy(), w)


def test_sparsegpt_validation():
    mod = linear(np.eye(4, dtype=np.float32))
    with pytest.raises(ValueError):
        prune_sparsegpt(mod)
    with pytest.raises(ValueError):
        prune_sparsegpt(mod, activations=np.zeros((4, 4)), damping=0.0)
    with pytest.raises(ValueError):
        prune_sparsegpt(mod, activations=np.ones((4, 4)), damping=-1.0)
    with pytest.raises(ValueError):
        prune_sparsegpt(mod, activations=np.ones((4, 4)), blocksize=0)


def test_nm_structured_magnitude_and_wanda():
    mod = linear([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    prune_nm_structured(mod, n=2, m=4)
    assert mod.weight.to_numpy().tolist() == [[8.0, 7.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0]]
    assert get_sparsity(mod.weight) == 0.5
    weighted = linear([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    norms = np.diag([1.0, 1.0, 10.0, 100.0, 1.0, 1.0, 1.0, 1.0])
    prune_nm_structured(weighted, n=2, m=4, activations=norms)
    assert weighted.weight.to_numpy().tolist() == [[0.0, 0.0, 6.0, 5.0, 4.0, 3.0, 0.0, 0.0]]
    noop = linear([[1.0, 2.0, 3.0, 4.0]])
    prune_nm_structured(noop, n=4, m=4)
    assert noop.weight.to_numpy().tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_nm_validation():
    mod = linear([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        prune_nm_structured(mod, n=2, m=4)
    with pytest.raises(ValueError):
        prune_nm_structured(linear([[1.0, 2.0, 3.0, 4.0]]), n=5, m=4)
    with pytest.raises(ValueError):
        prune_nm_structured(linear([[1.0, 2.0, 3.0, 4.0]]), n=2, m=4, activations=np.ones((2, 3)))


def test_block_influence():
    a = ox.from_numpy(np.array([[1.0, 0.0]], dtype=np.float32))
    assert block_influence(a, a) == pytest.approx(0.0)
    b = ox.from_numpy(np.array([[0.0, 1.0]], dtype=np.float32))
    assert block_influence(a, b) == pytest.approx(1.0)
    c = ox.from_numpy(np.array([[-1.0, 0.0]], dtype=np.float32))
    assert block_influence(a, c) == pytest.approx(2.0)
    rows = ox.from_numpy(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    outs = ox.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert block_influence(rows, outs) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        block_influence(a, ox.from_numpy(np.zeros((2, 2), dtype=np.float32)))


def test_select_layers_to_drop():
    assert select_layers_to_drop([0.5, 0.1, 0.9, 0.2], 2) == [1, 3]
    assert select_layers_to_drop([0.5, 0.1], 0) == []
    with pytest.raises(ValueError):
        select_layers_to_drop([0.5, 0.1], 2)
    with pytest.raises(ValueError):
        select_layers_to_drop([], 0)
    with pytest.raises(ValueError):
        select_layers_to_drop([0.1, float("nan")], 1)


def test_drop_sequential_blocks():
    model = ox.nn.Sequential([ox.nn.Linear(4, 4), ox.nn.ReLU(), ox.nn.Linear(4, 4)])
    assert len(model) == 3
    cut = drop_sequential_blocks(model, [1])
    assert isinstance(cut, ox.nn.Sequential)
    assert len(cut) == 2
    assert len(model) == 3
    out = cut(ox.from_numpy(np.ones((2, 4), dtype=np.float32)))
    assert list(out.shape) == [2, 4]
    last = drop_sequential_blocks(model, [-1])
    assert len(last) == 2
    same = drop_sequential_blocks(model, [])
    assert len(same) == 3
    with pytest.raises(ValueError):
        drop_sequential_blocks(model, [0, 1, 2])
    with pytest.raises(ValueError):
        drop_sequential_blocks(model, [5])
    with pytest.raises(ValueError):
        drop_sequential_blocks(ox.nn.Linear(4, 4), [0])


def test_remove_pruning_clears_new_masks():
    mod = linear([[1.0, 2.0, 3.0, 4.0]])
    prune_wanda(mod, activations=np.ones((2, 4)), amount=0.5)
    assert has_pruning_mask(mod)
    remove_pruning(mod)
    assert not has_pruning_mask(mod)
