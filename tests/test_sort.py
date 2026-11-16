"""Z25: sort/topk/searchsorted vs torch ground truth.

The native engine methods (stable, differentiable, i64-exact) serve both the
Tensor methods and, via validation shells in `primitives.py`, the module
functions. NaN placement, N-D searchsorted rows, and i64 past 2^53 are pinned.
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy


def test_sort_values_and_nan_both_directions():
    x = np.array([[3.0, 1.0, 2.0, np.nan], [0.5, -1.0, 4.0, 0.0]], dtype=np.float32)
    for desc in (False, True):
        ov, oi = T(x).sort(-1, descending=desc)
        tv, ti = th(x).sort(-1, descending=desc)
        np.testing.assert_allclose(ov.to_numpy(), tv.numpy(), equal_nan=True)
        np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())
        mv, mi = ox.sort(T(x), dim=-1, descending=desc)
        np.testing.assert_allclose(mv.to_numpy(), tv.numpy(), equal_nan=True)


def test_argsort_exact_without_ties_strided():
    a = np.array([[3.0, 1.0], [2.0, 0.0], [5.0, 4.0]], dtype=np.float32)
    oi = T(a).transpose(0, 1).argsort(-1).to_numpy()
    ti = th(a).transpose(0, 1).argsort(-1).numpy()
    np.testing.assert_array_equal(oi, ti)


def test_sort_ties_are_stable_and_consistent():
    t = np.array([2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    v, i = T(t).sort(-1)
    np.testing.assert_array_equal(i.to_numpy(), [1, 3, 0, 2, 4])
    v, i = T(t).sort(-1, descending=True)
    np.testing.assert_array_equal(i.to_numpy(), [0, 2, 4, 1, 3])
    # Gather consistency: values are exactly input[indices], sorted.
    vv = v.to_numpy()
    assert (np.diff(vv) <= 0).all()
    np.testing.assert_array_equal(vv, t[i.to_numpy()])


def test_topk_sorted_unsorted_and_dtypes():
    x = np.array([[3.0, 1.0, 2.0, np.nan], [0.5, -1.0, 4.0, 4.0]], dtype=np.float32)
    for k in (1, 2, 4):
        for largest in (True, False):
            ov, oi = T(x).topk(k, -1, largest=largest, sorted=True)
            tv, ti = th(x).topk(k, -1, largest=largest, sorted=True)
            np.testing.assert_allclose(ov.to_numpy(), tv.numpy(), equal_nan=True)
            ou, _ = T(x).topk(k, -1, largest=largest, sorted=False)
            tu, _ = th(x).topk(k, -1, largest=largest, sorted=False)
            np.testing.assert_allclose(np.sort(ou.to_numpy()), np.sort(tu.numpy()),
                                       equal_nan=True)
    for dt in (np.int32, np.int64, np.float64):
        a = np.array([[3, 1, 2], [0, -1, 5]], dtype=dt)
        ov, oi = T(a).topk(2, -1)
        tv, ti = th(a).topk(2, -1)
        np.testing.assert_array_equal(ov.to_numpy(), tv.numpy())
        np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())
    # Extension: torch topk rejects bool on CPU; oxideon sorts False < True.
    b = np.array([[True, False, True], [False, False, True]])
    ov, oi = T(b).topk(2, -1)
    np.testing.assert_array_equal(ov.to_numpy(), [[True, True], [True, False]])
    np.testing.assert_array_equal(oi.to_numpy(), [[0, 2], [2, 0]])


def test_topk_native_k0_and_i64_exact():
    v, i = T(np.array([3.0, 1.0])).topk(0, -1)
    assert v.to_numpy().size == 0 and i.to_numpy().size == 0
    big = np.array([2**53 + 1, 2**53, 2**53 - 1], dtype=np.int64)
    ov, oi = T(big).topk(3, -1, largest=False, sorted=True)
    tv, ti = th(big).topk(3, -1, largest=False, sorted=True)
    np.testing.assert_array_equal(ov.to_numpy(), tv.numpy())
    np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())
    assert oi.to_numpy().tolist() == [2, 1, 0]


def test_sort_topk_gradients_flow():
    xr = T(np.array([3.0, 1.0, 2.0], dtype=np.float32)).requires_grad_(True)
    xr.sort(-1)[0].sum().backward()
    np.testing.assert_allclose(xr.grad.numpy(), [1.0, 1.0, 1.0])
    xr = T(np.array([3.0, 1.0, 2.0], dtype=np.float32)).requires_grad_(True)
    xr.topk(2, -1)[0].sum().backward()
    np.testing.assert_allclose(xr.grad.numpy(), [1.0, 0.0, 1.0])


def test_searchsorted_sides_nan_empty_and_scalar():
    seq = np.array([1.0, 3.0, 3.0, 5.0, 8.0], dtype=np.float32)
    q = np.array([[-1.0, 1.0, 3.0, 4.0, 9.0, np.nan]], dtype=np.float32)
    for right in (False, True):
        got = ox.searchsorted(T(seq), T(q), right=right).to_numpy()
        want = torch.searchsorted(th(seq), th(q), right=right).numpy()
        np.testing.assert_array_equal(got, want)
    got = ox.searchsorted(T(np.array([], dtype=np.float32)), T(q)).to_numpy()
    want = torch.searchsorted(th(np.array([], dtype=np.float32)), th(q)).numpy()
    np.testing.assert_array_equal(got, want)
    # from_numpy presents 0-d input as 1-elem 1-D (pre-existing, out of
    # scope); the engine-level 0-d -> 0-d rule is pinned in Rust instead.
    s = ox.searchsorted(T(seq), T(np.array([4.0], dtype=np.float32))).to_numpy()
    np.testing.assert_array_equal(s, [3])


def test_searchsorted_nd_rows_broadcast_rule_and_errors():
    rng = np.random.default_rng(0)
    seq = np.sort(rng.standard_normal((2, 2, 5)).astype(np.float32), axis=-1)
    val = rng.standard_normal((2, 2, 3)).astype(np.float32)
    for right in (False, True):
        got = ox.searchsorted(T(seq), T(val), right=right).to_numpy()
        want = torch.searchsorted(th(seq), th(val), right=right).numpy()
        np.testing.assert_array_equal(got, want)
    bad = rng.standard_normal((2, 3)).astype(np.float32)
    with pytest.raises(RuntimeError):
        ox.searchsorted(T(seq), T(bad))
    with pytest.raises(RuntimeError):
        torch.searchsorted(th(seq), th(bad))
    with pytest.raises(ValueError, match="side"):
        ox.searchsorted(T(seq[0, 0]), T(val[0, 0]), side="middle")


def test_searchsorted_i64_mixed_and_out_int32():
    seq = np.array([2**53, 2**53 + 2, 2**53 + 4], dtype=np.int64)
    q = np.array([2**53 + 1, 2**53 + 3], dtype=np.int64)
    got = ox.searchsorted(T(seq), T(q)).to_numpy()
    want = torch.searchsorted(th(seq), th(q)).numpy()
    np.testing.assert_array_equal(got, want)
    assert got.tolist() == [1, 2]
    got = ox.searchsorted(T(np.array([1, 3, 5], dtype=np.int64)),
                          T(np.array([2.0], dtype=np.float32))).to_numpy()
    want = torch.searchsorted(th(np.array([1, 3, 5], dtype=np.int64)),
                              th(np.array([2.0], dtype=np.float32))).numpy()
    np.testing.assert_array_equal(got, want)
    o = ox.searchsorted(T(np.array([1.0, 2.0], dtype=np.float32)),
                        T(np.array([1.5], dtype=np.float32)), out_int32=True)
    assert o.to_numpy().dtype == np.int32
    np.testing.assert_array_equal(o.to_numpy(), [1])


def test_sort_module_dim_validation():
    x = T(np.array([3.0, 1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="out of range"):
        ox.sort(x, dim=5)
    with pytest.raises(ValueError, match="out of range"):
        ox.argsort(x, dim=-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
