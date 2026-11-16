"""Z39: mode vs torch ground truth.

The native engine kernel (per-slice counts, smallest-wins ties, NaN groups
and sorts smallest) serves both the Tensor method and, via pure delegation
in `primitives.py`, the module function (which keeps the `_ModeResult`
namedtuple). Values are pinned exactly against torch.

Deliberate divergence from torch (pinned below): on ties torch's indices
follow its internal unstable sort order (content-dependent: usually the
last occurrence, sometimes the first, e.g. `[1, nan, 1] -> 0`); oxideon
deterministically reports the FIRST occurrence (the shim's documented
contract). Values match torch everywhere, including NaN grouping,
smallest-wins ties, and zero-sign grouping (either sign compares equal).
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy


def test_mode_values_match_torch_ties_go_smallest():
    cases = [
        np.array([1.0, 3.0, 3.0, 1.0, 2.0], dtype=np.float32),
        np.array([2, 1, 2, 1]),
        np.array([5.0], dtype=np.float32),
        np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32),
        np.array([3.0, 1.0, 2.0, 1.0, 3.0, 2.0], dtype=np.float32),
    ]
    for x in cases:
        ov, oi = T(x).mode(dim=-1)
        tv, ti = torch.mode(th(x), -1)
        assert ov.to_numpy() == tv.numpy()
        # Index always addresses an occurrence of the reported value.
        assert x[oi.to_numpy()] == ov.to_numpy()
        mv = ox.mode(T(x)).values.to_numpy()
        assert mv == tv.numpy()


def test_mode_indices_are_first_occurrences():
    # torch reports content-dependent (usually last) occurrences; oxideon
    # pins the deterministic first occurrence instead.
    x = np.array([1.0, 3.0, 3.0, 1.0, 2.0], dtype=np.float32)
    ov, oi = ox.mode(T(x), dim=-1)
    assert ov.to_numpy() == 1.0
    assert oi.to_numpy() == 0
    assert torch.mode(th(x), -1).indices.item() == 3
    y = np.array([2, 1, 2, 1])
    assert ox.mode(T(y), dim=-1).indices.to_numpy() == 1


def test_mode_nan_groups_and_sorts_smallest():
    cases = [
        ([1.0, np.nan, 1.0], 1.0, 0),
        ([np.nan, np.nan, 1.0], np.nan, 0),
        ([np.nan, 1.0, 2.0], np.nan, 0),
        ([np.nan, 1.0, 1.0], 1.0, 1),
        ([1.0, 1.0, np.nan], 1.0, 0),
    ]
    for x, want_v, want_i in cases:
        a = np.array(x, dtype=np.float32)
        ov, oi = ox.mode(T(a), dim=-1)
        tv = torch.mode(th(a), -1).values.numpy()
        if np.isnan(want_v):
            assert np.isnan(ov.to_numpy()) and np.isnan(tv)
        else:
            assert ov.to_numpy() == tv == want_v
        assert oi.to_numpy() == want_i


def test_mode_zero_sign_groups():
    a = np.array([0.0, -0.0, 1.0, 1.0], dtype=np.float32)
    ov, oi = ox.mode(T(a), dim=-1)
    tv = torch.mode(th(a), -1).values.numpy()
    assert ov.to_numpy() == tv == 0.0
    assert oi.to_numpy() == 0
    b = np.array([0.0, -0.0, -0.0, 1.0], dtype=np.float32)
    assert ox.mode(T(b), dim=-1).values.to_numpy() == 0.0


def test_mode_2d_dims_and_keepdim():
    a = np.array([[1, 3, 3, 1, 2], [2, 2, 1, 1, 1]])
    for dim in (0, 1):
        for keepdim in (False, True):
            ov, oi = ox.mode(T(a), dim=dim, keepdim=keepdim)
            tv, ti = torch.mode(th(a), dim=dim, keepdim=keepdim)
            assert ov.to_numpy().shape == tv.numpy().shape
            np.testing.assert_array_equal(ov.to_numpy(), tv.numpy())
    ov, oi = ox.mode(T(a), dim=1)
    np.testing.assert_array_equal(oi.to_numpy(), [0, 2])


def test_mode_scalar_bool_and_big_int():
    v, i = ox.mode(T(np.array(5.0, dtype=np.float32)), dim=-1)
    assert v.to_numpy() == 5.0
    assert i.to_numpy() == 0
    assert v.to_numpy().shape == ()
    b = np.array([True, False, True])
    bv, bi = ox.mode(T(b), dim=-1)
    assert bv.to_numpy() == True
    assert bi.to_numpy() == 0
    assert torch.mode(th(b)).values.item() == True
    big = np.array([2**53 + 1, 2**53 + 1, 0], dtype=np.int64)
    vv, ii = ox.mode(T(big), dim=-1)
    assert vv.to_numpy() == 2**53 + 1
    assert ii.to_numpy() == 0


def test_mode_empty_and_bad_dim_raise():
    with pytest.raises(RuntimeError):
        ox.mode(T(np.empty((0,), dtype=np.float32)), dim=-1)
    with pytest.raises(IndexError):
        torch.mode(th(np.empty((0,), dtype=np.float32)), -1)
    with pytest.raises(RuntimeError):
        ox.mode(T(np.array([1, 2, 3])), dim=5)


def test_mode_grad_flows_to_reported_index():
    x = np.array([1.0, 3.0, 3.0, 1.0, 2.0], dtype=np.float32)
    o = T(x).requires_grad_(True)
    ox.mode(o, dim=-1).values.sum().backward()
    np.testing.assert_array_equal(o.grad.to_numpy(), [1.0, 0.0, 0.0, 0.0, 0.0])
    t = th(x).requires_grad_(True)
    torch.mode(t, -1).values.sum().backward()
    # torch's gradient follows its own (last) index; oxideon's follows first.
    np.testing.assert_array_equal(t.grad.numpy(), [0.0, 0.0, 0.0, 1.0, 0.0])
