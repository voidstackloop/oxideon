"""Tests for weight-space model merging (oxideon.merge)."""
import numpy as np
import pytest

import oxideon as ox
from oxideon import (
    average_merge,
    dare_merge,
    slerp_merge,
    task_arithmetic_merge,
    ties_merge,
)


def t(arr):
    return ox.from_numpy(np.asarray(arr, dtype=np.float32))


def state(w, b):
    return {"w": t(w), "b": t(b)}


def test_top_level_exports():
    assert ox.ties_merge is ties_merge
    assert ox.slerp_merge is slerp_merge
    assert ox.merge.dare_merge is dare_merge


def test_average_merge():
    out = average_merge([state([[0.0]], [0.0]), state([[10.0]], [4.0])])
    assert out["w"].to_numpy().tolist() == [[5.0]]
    assert out["b"].to_numpy().tolist() == [2.0]
    out = average_merge([state([[0.0]], [0.0]), state([[10.0]], [0.0])], weights=[1, 3])
    assert out["w"].to_numpy().tolist() == [[7.5]]
    single = state([[3.0]], [1.0])
    out = average_merge([single])
    assert out["w"].to_numpy().tolist() == [[3.0]]
    assert out["w"] is not single["w"]
    with pytest.raises(ValueError):
        average_merge([])
    with pytest.raises(ValueError):
        average_merge([state([[0.0]], [0.0])], weights=[1, 2])


def test_task_arithmetic_merge():
    base = state([[0.0, 0.0]], [0.0])
    fts = [state([[1.0, 1.0]], [1.0]), state([[3.0, 3.0]], [3.0])]
    out = task_arithmetic_merge(base, fts, weights=[1, 1])
    assert out["w"].to_numpy().tolist() == [[4.0, 4.0]]
    out = task_arithmetic_merge(base, fts)
    assert out["w"].to_numpy().tolist() == [[2.0, 2.0]]
    with pytest.raises(ValueError):
        task_arithmetic_merge(base, [])


def test_ties_merge_hand_computed():
    base = state([[0.0, 0.0], [0.0, 0.0]], [0.0])
    ft1 = state([[2.0, -1.0], [0.1, 0.0]], [1.0])
    ft2 = state([[1.0, 3.0], [0.0, 0.1]], [-1.0])
    out = ties_merge(base, [ft1, ft2])
    assert out["w"].to_numpy().flatten().tolist() == pytest.approx([1.5, 3.0, 0.1, 0.1])
    assert out["b"].to_numpy().tolist() == pytest.approx([1.0])
    single = ties_merge(base, [ft1], density=0.5)
    assert single["w"].to_numpy().flatten().tolist() == pytest.approx([2.0, -1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        ties_merge(base, [ft1], density=0.0)
    with pytest.raises(ValueError):
        ties_merge(base, [ft1], density=1.5)


def test_dare_merge_drop_zero_is_average_and_deterministic():
    base = state([[1.0, 2.0]], [0.5])
    fts = [state([[3.0, 0.0]], [1.5]), state([[1.0, 4.0]], [-0.5])]
    avg = average_merge(fts)
    dare = dare_merge(base, fts, drop_rate=0.0, seed=7)
    assert dare["w"].to_numpy().flatten().tolist() == pytest.approx([2.0, 2.0])
    assert dare["w"] is not avg["w"]
    again = dare_merge(base, fts, drop_rate=0.5, seed=3)
    twice = dare_merge(base, fts, drop_rate=0.5, seed=3)
    assert again["w"].to_numpy().tolist() == twice["w"].to_numpy().tolist()


def test_dare_merge_statistics():
    base = state(np.zeros(100), [0.0])
    ft = state(np.full(100, 2.0), [0.0])
    out = dare_merge(base, [ft], drop_rate=0.9, seed=0)
    vals = out["w"].to_numpy().flatten()
    assert set(np.unique(vals).tolist()) == {0.0, 20.0}
    frac_zero = float(np.mean(vals == 0.0))
    assert 0.8 < frac_zero < 1.0
    with pytest.raises(ValueError):
        dare_merge(base, [ft], drop_rate=1.0)
    with pytest.raises(ValueError):
        dare_merge(base, [ft], seed=-1)


def test_slerp_merge():
    a = state([[1.0, 0.0]], [0.0])
    b = state([[0.0, 1.0]], [0.0])
    assert slerp_merge(a, b, t=0.0)["w"].to_numpy().tolist() == [[1.0, 0.0]]
    assert slerp_merge(a, b, t=1.0)["w"].to_numpy().tolist() == [[0.0, 1.0]]
    mid = slerp_merge(a, b, t=0.5)["w"].to_numpy().flatten().tolist()
    assert mid == pytest.approx([0.7071068, 0.7071068], abs=1e-6)
    same = slerp_merge(a, a, t=0.3)["w"].to_numpy().flatten().tolist()
    assert same == pytest.approx([1.0, 0.0])
    zero = slerp_merge(state([[0.0, 0.0]], [0.0]), b, t=0.5)["w"].to_numpy().flatten().tolist()
    assert zero == pytest.approx([0.0, 0.5])
    with pytest.raises(ValueError):
        slerp_merge(a, b, t=2.0)


def test_merge_validation_and_immutability():
    base = state([[0.0]], [0.0])
    good = state([[1.0]], [1.0])
    bad_keys = {"w": t([[1.0]]), "z": t([1.0])}
    with pytest.raises(ValueError):
        average_merge([base, bad_keys])
    bad_shape = state([[1.0, 2.0]], [1.0])
    with pytest.raises(ValueError):
        average_merge([base, bad_shape])
    nan_state = state([[float("nan")]], [0.0])
    with pytest.raises(ValueError):
        average_merge([base, nan_state])
    with pytest.raises(ValueError):
        average_merge([base, good], weights=[1, float("nan")])
    before_w = base["w"].to_numpy().copy()
    before_g = good["w"].to_numpy().copy()
    average_merge([base, good])
    ties_merge(base, [good])
    dare_merge(base, [good], seed=1)
    slerp_merge(base, good)
    task_arithmetic_merge(base, [good])
    assert np.array_equal(base["w"].to_numpy(), before_w)
    assert np.array_equal(good["w"].to_numpy(), before_g)
